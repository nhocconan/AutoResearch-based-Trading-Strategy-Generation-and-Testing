#!/usr/bin/env python3
"""
1d_RSI2_MeanReversion_WeeklyTrend
Hypothesis: Use 2-period RSI on daily timeframe for mean reversion entries, filtered by weekly trend (EMA50) and volume confirmation.
Works in both bull and bear markets: in uptrend, buy RSI2 dips; in downtrend, sell RSI2 bounces.
Designed for 10-20 trades/year to avoid fee drag while capturing mean reversion at trend-aligned levels.
"""

name = "1d_RSI2_MeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Daily price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 2-period RSI for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (50), RSI (2)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if weekly trend is not available
        if np.isnan(ema_50_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend AND RSI2 oversold (<10) with volume confirmation
            if close[i] > ema_50_weekly_aligned[i] and rsi[i] < 10 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend AND RSI2 overbought (>90) with volume confirmation
            elif close[i] < ema_50_weekly_aligned[i] and rsi[i] > 90 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI2 overbought (>80) or trend turns down
            if rsi[i] > 80 or close[i] < ema_50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI2 oversold (<20) or trend turns up
            if rsi[i] < 20 or close[i] > ema_50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals