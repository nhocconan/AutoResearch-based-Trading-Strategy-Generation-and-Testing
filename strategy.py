#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with volume spike and 1w EMA34 trend filter.
# Long when price breaks above upper Donchian + volume spike + price > 1w EMA34
# Short when price breaks below lower Donchian + volume spike + price < 1w EMA34
# Exit when price crosses back through Donchian midpoint or volume drops below 80% of average.
# Uses daily timeframe to reduce trade frequency, avoid fee drag, and capture longer trends.
# Target: 15-30 trades/year to stay within optimal range for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    # Load 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d Donchian channels (20-day high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper (20-day high) and lower (20-day low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume spike filter (20-day average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        ema34 = ema34_1w_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + volume spike + price > 1w EMA34
            if price > upper and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + volume spike + price < 1w EMA34
            elif price < lower and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through midpoint or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below midpoint or volume dries up
                if price < mid or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above midpoint or volume dries up
                if price > mid or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0