#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + 1w EMA34 Trend
# Long when Jaw < Teeth < Lips (bullish alignment) + price > Lips + 1d volume > 2x 20-day avg
# Short when Jaw > Teeth > Lips (bearish alignment) + price < Jaw + 1d volume > 2x 20-day avg
# Exit when Alligator lines cross or price crosses Jaw/Lips
# Williams Alligator uses SMAs: Jaw=SMA13(8), Teeth=SMA8(5), Lips=SMA5(3)
# Trend filter: price > 1w EMA34 for longs, price < 1w EMA34 for shorts
# Volume confirms conviction, works in both bull (strong alignment) and bear (strong alignment) regimes
# Target: 12-37 trades/year by requiring strong alignment + volume spike + trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Williams Alligator (SMA-based)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # SMA13
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # SMA8
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values   # SMA5
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after Alligator warmup
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
        ema34_val = ema34_1w_aligned[i]
        
        # Get current 1d volume (12h bar = 0.5 days, so use same day's volume)
        day_idx = i // 2  # 2 bars per day (24h/12h)
        if day_idx >= len(df_1d):
            day_idx = len(df_1d) - 1
        volume = df_1d['volume'].iloc[day_idx] if day_idx >= 0 else df_1d['volume'].iloc[0]
        
        # Volume confirmation: current 1d volume > 2x 20-day average
        volume_confirm = volume > 2.0 * vol_ma if not np.isnan(vol_ma) else False
        
        # Alligator alignment signals
        bullish_aligned = jaw_val < teeth_val and teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_aligned = jaw_val > teeth_val and teeth_val > lips_val  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Bullish alignment + price > Lips + above 1w EMA34 + volume confirmation
            if bullish_aligned and price > lips_val and price > ema34_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < Jaw + below 1w EMA34 + volume confirmation
            elif bearish_aligned and price < jaw_val and price < ema34_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Alligator loses bullish alignment or price crosses below Jaw
                if not bullish_aligned or price < jaw_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Alligator loses bearish alignment or price crosses above Lips
                if not bearish_aligned or price > lips_val:
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