#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R (14) mean reversal with 1d EMA50 trend filter and volume confirmation.
- Williams %R < -80 = oversold (long setup), > -20 = overbought (short setup)
- Entry requires: Williams %R extreme + price reversal candle + volume spike + 1d EMA50 trend alignment
- Exit: Williams %R returns to -50 (mean reversion midpoint) or opposite extreme
- Uses 4h timeframe (primary) and 1d HTF for EMA50 trend (proven BTC/ETH edge from DB)
- Volume spike: current 4h volume > 1.8 * 20-period volume MA (balanced to avoid overtrading)
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
- Works in both bull/bear: trend filter avoids counter-trend trades, Williams %R captures reversals in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Trend filter: 4h close vs 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Price reversal candle: bullish engulfing for long, bearish engulfing for short
    bullish_engulfing = (close > open_) & (open_ < close_) & (close > open_.shift(1)) & (open_ < close_.shift(1))
    bearish_engulfing = (close < open_) & (open_ > close_) & (close < open_.shift(1)) & (open_ > close_.shift(1))
    # Fix: define open_ from prices
    open_ = prices['open'].values
    bullish_engulfing = (close > open_) & (open_ < close_) & (close > np.roll(open_, 1)) & (open_ < np.roll(close_, 1))
    bearish_engulfing = (close < open_) & (open_ > close_) & (close < np.roll(open_, 1)) & (open_ > np.roll(close_, 1))
    # Handle first bar
    bullish_engulfing[0] = False
    bearish_engulfing[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need 1d EMA50, Williams %R (14), and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND bullish engulfing AND uptrend AND volume spike
            if williams_r[i] < -80 and bullish_engulfing[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND bearish engulfing AND downtrend AND volume spike
            elif williams_r[i] > -20 and bearish_engulfing[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or overbought (> -20)
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or oversold (< -80)
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_14_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0