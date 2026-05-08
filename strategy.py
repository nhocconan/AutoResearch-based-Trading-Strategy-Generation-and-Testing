#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI_Pullback_MultiTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on daily closes
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 4h data for trend filters (EMA20 and EMA50)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_ema20 = (close > ema20).astype(float)
    trend_ema50 = (close > ema50).astype(float)
    trend_combined = (trend_ema20 + trend_ema50) / 2.0  # Average of both trends
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(trend_combined[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: RSI pullback from oversold (<30) with bullish alignment and volume
            long_cond = (rsi_1d_aligned[i] < 30 and trend_combined[i] > 0.5 and vol_filter[i])
            
            # Short entry: RSI pullback from overbought (>70) with bearish alignment and volume
            short_cond = (rsi_1d_aligned[i] > 70 and trend_combined[i] < 0.5 and vol_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI reaches overbought (>70) or trend turns bearish
            if rsi_1d_aligned[i] > 70 or trend_combined[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI reaches oversold (<30) or trend turns bullish
            if rsi_1d_aligned[i] < 30 or trend_combined[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI pullback strategy on 4h timeframe using daily RSI for overbought/oversold signals.
# Uses dual EMA trend filter (20/50) for multi-timeframe alignment and volume confirmation.
# Enters long when daily RSI < 30 (oversold) with bullish trend alignment and volume spike.
# Enters short when daily RSI > 70 (overbought) with bearish trend alignment and volume spike.
# Exits when RSI reverses or trend deteriorates. Designed to work in both bull and bear markets
# by capturing mean-reversion moves within the prevailing trend. Discrete sizing (0.25) minimizes
# churn. Target: 25-40 trades/year. Works in bull markets by buying pullbacks in uptrends and
# in bear markets by selling rallies in downtrends.