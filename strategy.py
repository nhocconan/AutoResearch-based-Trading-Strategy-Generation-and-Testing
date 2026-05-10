#!/usr/bin/env python3
# 1D_RSI20_EMA50_TREND
# Hypothesis: RSI(20) combined with EMA50 trend filter provides mean-reversion signals in ranging markets and trend-following signals in trending markets. The strategy uses weekly trend filter to avoid counter-trend trades, with volatility-adjusted position sizing to manage risk. Works in both bull and bear markets by adapting to regime via weekly EMA slope and daily RSI extremes.

name = "1D_RSI20_EMA50_TREND"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA for trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(20)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=20, adjust=False, min_periods=20).mean()
    avg_loss = pd.Series(loss).ewm(span=20, adjust=False, min_periods=20).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily EMA50 for dynamic support/resistance
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility normalization (ATR-like using 20-day true range)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(ema_50[i]) or np.isnan(atr_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade in direction of weekly EMA slope
        weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
        weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        
        # Volatility-adjusted position size (base 0.25, scaled by ATR)
        base_size = 0.25
        vol_factor = np.clip(0.5 / (atr_20[i] / close[i] + 0.01), 0.5, 2.0)  # Normalize volatility
        position_size = base_size * vol_factor
        position_size = min(0.35, position_size)  # Cap at 0.35
        
        if position == 0:
            # Long conditions: RSI oversold in uptrend OR EMA bounce in downtrend
            if (rsi[i] < 30 and weekly_uptrend) or (close[i] > ema_50[i] and close[i-1] <= ema_50[i-1] and weekly_uptrend):
                signals[i] = position_size
                position = 1
            # Short conditions: RSI overbought in downtrend OR EMA rejection in uptrend
            elif (rsi[i] > 70 and weekly_downtrend) or (close[i] < ema_50[i] and close[i-1] >= ema_50[i-1] and weekly_downtrend):
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: RSI overbought OR price breaks below EMA50 in downtrend
            if rsi[i] > 70 or (close[i] < ema_50[i] and weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: RSI oversold OR price breaks above EMA50 in uptrend
            if rsi[i] < 30 or (close[i] > ema_50[i] and weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals