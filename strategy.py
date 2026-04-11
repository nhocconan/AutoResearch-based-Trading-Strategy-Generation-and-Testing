#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (HMA(21)) and volume confirmation.
# Long when price breaks above Donchian(20) high + HMA(21) uptrend + volume > 1.5x avg.
# Short when price breaks below Donchian(20) low + HMA(21) downtrend + volume > 1.5x avg.
# Exit when price crosses back through Donchian(20) midline or opposite breakout occurs.
# Designed for 20-50 trades/year on 4h timeframe with focus on strong trending moves.
# Volume filter ensures breakouts have conviction, reducing false signals in chop.

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.average(x, weights=np.arange(1, half_len + 1)), raw=True
    ).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, 22)), raw=True
    ).values
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.average(x, weights=np.arange(1, sqrt_len + 1)), raw=True
    ).values
    
    # Align 12h HMA to 4h timeframe
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 12h trend direction from HMA slope
        if i > 20:
            hma_rising = hma_21_12h_aligned[i] > hma_21_12h_aligned[i-1]
            hma_falling = hma_21_12h_aligned[i] < hma_21_12h_aligned[i-1]
        else:
            hma_rising = hma_falling = False
        
        # Entry conditions
        bullish_breakout = close[i] > donchian_high[i]
        bearish_breakout = close[i] < donchian_low[i]
        
        bullish_entry = bullish_breakout and hma_rising and vol_filter
        bearish_entry = bearish_breakout and hma_falling and vol_filter
        
        # Exit conditions: opposite breakout or cross midline
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
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