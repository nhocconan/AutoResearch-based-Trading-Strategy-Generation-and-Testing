#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1w EMA34 trend filter and volume confirmation
# Uses 1h Elder Ray (Bull/Bear Power) to measure trend strength relative to EMA13
# 1w EMA34 defines the primary trend direction (avoids counter-trend trades)
# Volume spike (>1.8x 50-bar average) confirms institutional participation
# Works in both bull/bear: Elder Ray captures momentum, 1w trend filter avoids whipsaw, volume ensures validity
# Target: 80-160 total trades over 4 years (20-40/year) with discrete sizing 0.25

name = "6h_ElderRay_1wEMA34_VolumeSpike_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1w) < 34 or len(df_1h) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1w EMA34 trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1h EMA13 for Elder Ray
    close_1h_series = pd.Series(close_1h)
    ema13_1h = close_1h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1h - ema13_1h
    bear_power = low_1h - ema13_1h
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>1.8x 50-bar average)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.8 * vol_ma_50)
    
    # Align HTF indicators to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    bull_power_aligned = align_htf_to_ltf(prices, df_1h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1h, bear_power)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) AND uptrend (close > 1w EMA34) AND volume spike
            if bull_power_aligned[i] > 0 and close[i] > ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short: Bear Power < 0 (strong selling pressure) AND downtrend (close < 1w EMA34) AND volume spike
            elif bear_power_aligned[i] < 0 and close[i] < ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 40% of ATR from extreme (wider stop for 6h volatility)
            if close[i] <= long_extreme - 0.4 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 40% of ATR from extreme
            if close[i] >= short_extreme + 0.4 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals