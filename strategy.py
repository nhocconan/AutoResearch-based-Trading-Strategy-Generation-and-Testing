#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Uses 4h Supertrend for trend direction (HTF signal direction)
# - Uses 1h Camarilla pivot levels for entry timing (LTF precision)
# - Long when 4h Supertrend = uptrend AND price breaks above H3 resistance AND session 08-20 UTC
# - Short when 4h Supertrend = downtrend AND price breaks below L3 support AND session 08-20 UTC
# - Exit when price crosses opposite pivot level (H3 for shorts, L3 for longs)
# - Discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots work well in ranging markets; Supertrend filters for directional bias
# - Session filter avoids low-volume Asian session noise

name = "1h_4h_camarilla_supertrend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute session filter (08-20 UTC) - compute once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 4h Supertrend (ATR=10, mult=3) for trend direction
    hl2_4h = (df_4h['high'] + df_4h['low']) / 2
    atr_4h = pd.Series(0.0, index=df_4h.index)
    tr_4h = pd.Series(0.0, index=df_4h.index)
    tr_4h.iloc[0] = df_4h['high'].iloc[0] - df_4h['low'].iloc[0]
    for i in range(1, len(df_4h)):
        tr_4h.iloc[i] = max(
            df_4h['high'].iloc[i] - df_4h['low'].iloc[i],
            abs(df_4h['high'].iloc[i] - df_4h['close'].iloc[i-1]),
            abs(df_4h['low'].iloc[i] - df_4h['close'].iloc[i-1])
        )
    atr_4h = tr_4h.ewm(alpha=1/10, adjust=False, min_periods=10).mean()
    
    upper_band_4h = hl2_4h + 3 * atr_4h
    lower_band_4h = hl2_4h - 3 * atr_4h
    
    supertrend_4h = np.zeros(len(df_4h), dtype=float)
    direction_4h = np.ones(len(df_4h), dtype=int)  # 1=uptrend, -1=downtrend
    
    for i in range(1, len(df_4h)):
        if close_4h := df_4h['close'].iloc[i] > upper_band_4h.iloc[i-1]:
            direction_4h[i] = 1
        elif close_4h < lower_band_4h.iloc[i-1]:
            direction_4h[i] = -1
        else:
            direction_4h[i] = direction_4h[i-1]
            if direction_4h[i] == 1 and hl2_4h.iloc[i] < upper_band_4h.iloc[i-1]:
                upper_band_4h.iloc[i] = hl2_4h.iloc[i]
            if direction_4h[i] == -1 and hl2_4h.iloc[i] > lower_band_4h.iloc[i-1]:
                lower_band_4h.iloc[i] = hl2_4h.iloc[i]
        
        if direction_4h[i] == 1:
            supertrend_4h[i] = lower_band_4h.iloc[i]
        else:
            supertrend_4h[i] = upper_band_4h.iloc[i]
    
    # Align 4h Supertrend direction to 1h timeframe
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h.values)
    
    # Pre-compute 1h Camarilla pivot levels (using previous day's OHLC)
    # For intraday, we use rolling 24-period (24h = 1 day in 1h timeframe)
    lookback = 24
    if len(prices) < lookback:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous 24 periods
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla pivot calculations
    range_val = high_max - low_min
    camarilla_h3 = close_prev + (range_val * 1.1 / 4)
    camarilla_l3 = close_prev - (range_val * 1.1 / 4)
    camarilla_h4 = close_prev + (range_val * 1.1 / 2)
    camarilla_l4 = close_prev - (range_val * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(direction_4h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Check session filter
        if not session_filter[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: 4h uptrend AND price breaks above H3 resistance
            if (direction_4h_aligned[i] == 1 and 
                close[i] > camarilla_h3[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: 4h downtrend AND price breaks below L3 support
            elif (direction_4h_aligned[i] == -1 and 
                  close[i] < camarilla_l3[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses opposite pivot level
            exit_long = (position == 1 and close[i] < camarilla_l3[i])
            exit_short = (position == -1 and close[i] > camarilla_h3[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals