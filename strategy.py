#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKeltnerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for weekly Keltner channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA20 for middle line
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly ATR(10) for Keltner width
    tr_w = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                                 np.abs(low_1w - np.roll(close_1w, 1))))
    tr_w[0] = high_1w[0] - low_1w[0]
    atr10_1w = pd.Series(tr_w).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr10_1w)
    
    # Weekly Keltner channels: upper = EMA20 + 2*ATR10, lower = EMA20 - 2*ATR10
    keltner_upper = ema20_1w + 2 * atr10_1w
    keltner_lower = ema20_1w - 2 * atr10_1w
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # Weekly trend filter: EMA50 slope
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_slope = np.diff(ema50_1w_aligned, prepend=ema50_1w_aligned[0])
    
    # Daily volume filter: volume > 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema50_slope[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Keltner upper AND weekly EMA50 rising AND volume above average
            long_cond = (close[i] > keltner_upper_aligned[i] and 
                        ema50_slope[i] > 0 and
                        volume[i] > vol_ma20[i])
            
            # Short: Price breaks below weekly Keltner lower AND weekly EMA50 falling AND volume above average
            short_cond = (close[i] < keltner_lower_aligned[i] and 
                         ema50_slope[i] < 0 and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below weekly Keltner middle OR weekly EMA50 turns down
            if close[i] < ema20_1w_aligned[i] or ema50_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above weekly Keltner middle OR weekly EMA50 turns up
            if close[i] > ema20_1w_aligned[i] or ema50_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Keltner channel breakout with weekly EMA50 trend filter and volume confirmation.
# Works in bull markets via breakout continuation above upper Keltner band.
# Works in bear via mean reversion from lower Keltner band when trend turns down.
# Weekly timeframe reduces noise, daily entries provide timely execution.
# Target: 15-25 trades/year to minimize fee drag. Discrete sizing (0.25) reduces churn.