#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 1d volume confirmation + 1w trend filter
# Camarilla levels (R3/S3, R4/S4) from daily pivots provide institutional support/resistance
# Breakout of R4/S4 with volume confirmation indicates strong momentum continuation
# Weekly trend filter (price vs 200-period EMA) ensures alignment with higher timeframe momentum
# Works in bull/bear: weekly trend defines regime, Camarilla breakouts capture acceleration
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First value has no previous
    
    # Camarilla calculation: based on previous day's range
    rang = prev_high - prev_low
    # R3/S3 and R4/S4 levels
    r3 = prev_close + rang * 1.1 / 4
    s3 = prev_close - rang * 1.1 / 4
    r4 = prev_close + rang * 1.1 / 2
    s4 = prev_close - rang * 1.1 / 2
    
    # 1d average volume (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    close_s_1w = pd.Series(close_1w)
    ema200_1w = close_s_1w.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1d indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Align 1w trend to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 1d average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Weekly trend: bullish if price > EMA200, bearish if price < EMA200
        weekly_bullish = close[i] > ema200_1w_aligned[i]
        weekly_bearish = close[i] < ema200_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below R3 (profit taker) OR weekly trend turns bearish
            if close[i] < r3_aligned[i] or not weekly_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 (profit taker) OR weekly trend turns bullish
            if close[i] > s3_aligned[i] or not weekly_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakouts with volume confirmation and weekly trend alignment
            if weekly_bullish:
                # In weekly uptrend: look for long breakouts above R4
                if close[i] > r4_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
            elif weekly_bearish:
                # In weekly downtrend: look for short breakdowns below S4
                if close[i] < s4_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals