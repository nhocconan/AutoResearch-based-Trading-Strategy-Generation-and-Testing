#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily ATR filter and volume confirmation.
# Enter long when price breaks above weekly Donchian high and daily ATR > 20-day ATR average.
# Enter short when price breaks below weekly Donchian low and daily ATR > 20-day ATR average.
# Uses weekly Donchian channels (20 periods) and daily ATR (14 periods).
# Designed for 15-30 trades/year on 1d timeframe with focus on trend continuation.
# Volume filter ensures breakouts have conviction, reducing false signals.
# ATR filter avoids breakouts during low volatility periods.

name = "1d_1w_donchian_atr_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate daily ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day ATR average for filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_ma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Filters: volatility and volume
        vol_filter = atr_14[i] > atr_ma_20[i]  # ATR above average = sufficient volatility
        volume_filter = volume[i] > vol_ma_20[i]  # Volume above average
        
        # Entry conditions
        bullish_breakout = close[i] > donchian_high_20_aligned[i]
        bearish_breakout = close[i] < donchian_low_20_aligned[i]
        
        bullish_entry = bullish_breakout and vol_filter and volume_filter
        bearish_entry = bearish_breakout and vol_filter and volume_filter
        
        # Exit conditions: opposite breakout
        exit_long = close[i] < donchian_low_20_aligned[i]
        exit_short = close[i] > donchian_high_20_aligned[i]
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals