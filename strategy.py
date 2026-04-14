# 42246 | 4h_1d_camarilla_breakout_v25 (adapted for 4h)
# Hypothesis: Camarilla pivot levels from 1d (prior day's OHLC) act as strong support/resistance.
# Price breaking above/below R4/S4 with volume confirmation and 4h trend filter (EMA20) captures institutional breakouts.
# Works in bull/bear: trend filter avoids counter-trend trades; volume avoids false breakouts.
# Target: 20-40 trades/year, low frequency to avoid fee drag.
# Only long/short when price > EMA20 (bull) or < EMA20 (bear) to align with trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot points (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels using prior day's OHLC
    # R4 = close + 1.5 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # R2 = close + 0.6 * (high - low)
    # S2 = close - 0.6 * (high - low)
    # R1 = close + 0.3 * (high - low)
    # S1 = close - 0.3 * (high - low)
    # Pivot = (high + low + close) / 3
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels
    r4 = prev_close + 1.5 * (prev_high - prev_low)
    s4 = prev_close - 1.5 * (prev_high - prev_low)
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    r2 = prev_close + 0.6 * (prev_high - prev_low)
    s2 = prev_close - 0.6 * (prev_high - prev_low)
    r1 = prev_close + 0.3 * (prev_high - prev_low)
    s1 = prev_close - 0.3 * (prev_high - prev_low)
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(20) for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for 20-period EMA + 1-day lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above daily R4 AND above 4h EMA20 with volume confirmation
            if price > r4_aligned[i] and price > ema_20_4h_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below daily S4 AND below 4h EMA20 with volume confirmation
            elif price < s4_aligned[i] and price < ema_20_4h_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below daily S3 (strong support)
            if price < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above daily R3 (strong resistance)
            if price > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_camarilla_breakout_v25"
timeframe = "4h"
leverage = 1.0