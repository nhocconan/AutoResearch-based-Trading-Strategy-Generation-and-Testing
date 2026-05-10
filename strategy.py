#!/usr/bin/env python3
# 1h_4h1d_Momentum_Squeeze_Trend_Filter
# Hypothesis: Combines 4h EMA34 trend filter with 1d Bollinger Band squeeze (low volatility) 
# and 1h RSI(2) extreme reversals for precise entries. Works in bull/bear by only taking 
# trades aligned with 4h trend during low-volatility periods, reducing false breakouts.
# Target: 20-40 trades/year to minimize fee drag on 1h timeframe.

name = "1h_4h1d_Momentum_Squeeze_Trend_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # 4h trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_4h_up = close_4h > ema34_4h
    trend_4h_down = close_4h < ema34_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d Bollinger Band squeeze (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_1d + 2 * std20_1d
    lower_bb = sma20_1d - 2 * std20_1d
    bb_width = (upper_bb - lower_bb) / sma20_1d
    # Squeeze when BB width is below 20-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Align squeeze to 1h
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))
    
    # 1h RSI(2) for mean reversion entries
    rsi_period = 2
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # RSI < 10 = oversold, RSI > 90 = overbought
    rsi_oversold = rsi < 10
    rsi_overbought = rsi > 90
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold + 4h uptrend + volatility squeeze
            if (rsi_oversold[i] and
                trend_4h_up_aligned[i] > 0.5 and
                squeeze_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + 4h downtrend + volatility squeeze
            elif (rsi_overbought[i] and
                  trend_4h_down_aligned[i] > 0.5 and
                  squeeze_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI > 50 (mean reversion complete) or 4h trend turns down
            if (rsi[i] > 50 or
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI < 50 (mean reversion complete) or 4h trend turns up
            if (rsi[i] < 50 or
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals