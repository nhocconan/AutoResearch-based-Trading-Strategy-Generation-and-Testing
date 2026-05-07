#!/usr/bin/env python3
name = "1d_RSI_MeanReversion_TrendFilter"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(14, 200)
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price above weekly EMA200 (uptrend)
            if rsi[i] < 30 and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price below weekly EMA200 (downtrend)
            elif rsi[i] > 70 and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 50 (mean reversion) or stop loss
            if rsi[i] > 50 or close[i] < ema200_1w_aligned[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 (mean reversion) or stop loss
            if rsi[i] < 50 or close[i] > ema200_1w_aligned[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI mean reversion works in both bull and bear markets when filtered by weekly trend.
# In uptrends (price > weekly EMA200), buy oversold dips (RSI < 30).
# In downtrends (price < weekly EMA200), sell overbought rallies (RSI > 70).
# Exit on mean reversion (RSI crosses 50) or stop loss (2x ATR).
# Weekly trend filter reduces whipsaws and aligns with higher timeframe momentum.
# Position size 0.25 limits risk. Target ~15-25 trades/year to minimize fee drag.