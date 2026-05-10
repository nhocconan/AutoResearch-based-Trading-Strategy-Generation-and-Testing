#!/usr/bin/env python3
# 1d_Stochastic_RSI_MeanReversion
# Hypothesis: Mean reversion on daily timeframe using Stochastic RSI overbought/oversold levels
# combined with 1-week trend filter. Works in both bull/bear markets: Stochastic RSI identifies
# extreme conditions while weekly trend ensures trades align with higher timeframe momentum.
# Target: 15-25 trades/year (60-100 total over 4 years).

name = "1d_Stochastic_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Stochastic RSI calculation (14-period RSI, then stochastic of RSI)
    rsi_period = 14
    stoch_period = 14
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_series = pd.Series(rsi)
    stoch_rsi = (rsi_series - rsi_series.rolling(window=stoch_period, min_periods=stoch_period).min()) / \
                (rsi_series.rolling(window=stoch_period, min_periods=stoch_period).max() - 
                 rsi_series.rolling(window=stoch_period, min_periods=stoch_period).min() + 1e-10) * 100
    stoch_rsi = stoch_rsi.values
    
    # 1-week trend filter (EMA50 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1-week trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(stoch_rsi[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Stochastic RSI oversold (<20) and 1-week uptrend
            if (stoch_rsi[i] < 20 and
                trend_1w_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Stochastic RSI overbought (>80) and 1-week downtrend
            elif (stoch_rsi[i] > 80 and
                  trend_1w_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Stochastic RSI crosses above 50 (mean reversion complete) or trend turns down
            if (stoch_rsi[i] > 50 or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Stochastic RSI crosses below 50 (mean reversion complete) or trend turns up
            if (stoch_rsi[i] < 50 or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals