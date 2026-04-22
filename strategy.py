#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA trend + volume confirmation
# Long when Green > Red > Blue (bullish alignment) and price > 1d EMA34 and volume spike
# Short when Green < Red < Blue (bearish alignment) and price < 1d EMA34 and volume spike
# Exit when alignment breaks or volume drops
# Designed for low trade frequency (~20-40/year) with trend-following edge
# Works in both bull (strong uptrends) and bear (strong downtrends) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (using median price)
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw (Blue line) - 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (Red line) - 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (Green line) - 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        # Alligator alignment
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: Bullish alignment, price > 1d EMA34, volume spike
            if bullish_alignment and price > ema_34_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish alignment, price < 1d EMA34, volume spike
            elif bearish_alignment and price < ema_34_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alignment breaks or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or volume drops
                if not bullish_alignment or vol < vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or volume drops
                if not bearish_alignment or vol < vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0