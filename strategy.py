#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + 1w EMA34 Trend
# Long when Jaw < Teeth < Lips (bullish alignment), 1d volume > 2.0x 20-day average, price > 1w EMA34
# Short when Jaw > Teeth > Lips (bearish alignment), 1d volume > 2.0x 20-day average, price < 1w EMA34
# Williams Alligator uses SMAs: Jaw=13, Teeth=8, Lips=5 (all smoothed)
# Works in both bull (strong bullish alignment) and bear (strong bearish alignment)
# Volume confirms conviction, weekly EMA filters trend direction
# Target: 12-37 trades/year by requiring strict alignment + volume spike

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator components (1d timeframe)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (13-period SMMA, smoothed with 8-period shift)
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (8-period SMMA, smoothed with 5-period shift)
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips (5-period SMMA, smoothed with 3-period shift)
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Price and volume arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        # Get current 1d volume (12h = 0.5 day, so 2 bars per day)
        day_idx = i // 2
        if day_idx < len(df_1d):
            volume = df_1d['volume'].iloc[day_idx]
        else:
            volume = df_1d['volume'].iloc[-1] if len(df_1d) > 0 else 0
        
        # Volume confirmation: current 1d volume > 2.0x 20-day average
        volume_confirm = volume > 2.0 * vol_ma if vol_ma > 0 else False
        
        # Williams Alligator alignments
        bullish_align = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_align = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Bullish alignment, volume confirmation, price > weekly EMA34
            if bullish_align and volume_confirm and price > ema34_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment, volume confirmation, price < weekly EMA34
            elif bearish_align and volume_confirm and price < ema34_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if bullish alignment breaks or price crosses below weekly EMA34
                if not bullish_align or price < ema34_1w_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if bearish alignment breaks or price crosses above weekly EMA34
                if not bearish_align or price > ema34_1w_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_1wEMA34_Trend"
timeframe = "12h"
leverage = 1.0