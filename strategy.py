#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Supertrend (ATR=10, multiplier=3.0)
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=10, min_periods=10).mean()
    
    # Upper and Lower Bands
    hl2 = (high_1w + low_1w) / 2
    upper_band_1w = hl2 + (3.0 * atr_1w)
    lower_band_1w = hl2 - (3.0 * atr_1w)
    
    # Supertrend calculation
    supertrend_1w = np.zeros(len(close_1w))
    direction_1w = np.ones(len(close_1w))  # 1 = uptrend, -1 = downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w.iloc[i] > upper_band_1w.iloc[i-1]:
            direction_1w[i] = 1
        elif close_1w.iloc[i] < lower_band_1w.iloc[i-1]:
            direction_1w[i] = -1
        else:
            direction_1w[i] = direction_1w[i-1]
            if direction_1w[i] == 1 and lower_band_1w.iloc[i] < lower_band_1w.iloc[i-1]:
                lower_band_1w.iloc[i] = lower_band_1w.iloc[i-1]
            if direction_1w[i] == -1 and upper_band_1w.iloc[i] > upper_band_1w.iloc[i-1]:
                upper_band_1w.iloc[i] = upper_band_1w.iloc[i-1]
        
        if direction_1w[i] == 1:
            supertrend_1w[i] = lower_band_1w.iloc[i]
        else:
            supertrend_1w[i] = upper_band_1w.iloc[i]
    
    # Align Supertrend to daily timeframe
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    uptrend = close > supertrend_1w_aligned  # Price above Supertrend = uptrend
    downtrend = close < supertrend_1w_aligned  # Price below Supertrend = downtrend
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(supertrend_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: opposite Donchian break
        exit_long = close[i] < donchian_low[i-1]
        exit_short = close[i] > donchian_high[i-1]
        
        if position == 1:  # Long position
            # Exit on breakdown or trend reversal
            if exit_long or not uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on breakout or trend reversal
            if exit_short or not downtrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: upward breakout + uptrend + volume confirmation
            if breakout_up and uptrend[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: downward breakout + downtrend + volume confirmation
            elif breakout_down and downtrend[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals