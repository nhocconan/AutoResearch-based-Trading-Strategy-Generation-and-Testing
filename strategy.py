#!/usr/bin/env python3
name = "6h_7x14_RSIFailure_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(7) - fast momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/7, adjust=False, min_periods=7).mean()
    avg_loss = loss_series.ewm(alpha=1/7, adjust=False, min_periods=7).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi7 = (100 - (100 / (1 + rs))).values
    
    # RSI(14) - slower momentum for confirmation
    gain14 = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss14 = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs14 = gain14 / (loss14 + 1e-10)
    rsi14 = (100 - (100 / (1 + rs14))).values
    
    # 14-period EMA for pullback context
    ema14 = pd.Series(close).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly trend filter (1w SMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    weekly_uptrend = close > sma50_1w_aligned
    
    # Volume filter (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 20)  # ensure indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(rsi7[i]) or np.isnan(rsi14[i]) or np.isnan(ema14[i]) or np.isnan(weekly_uptrend[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI7 < 30 (oversold) AND RSI14 < 50 (weak momentum) AND price > EMA14 (pullback in uptrend) AND weekly uptrend AND volume
            if rsi7[i] < 30 and rsi14[i] < 50 and close[i] > ema14[i] and weekly_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI7 > 70 (overbought) AND RSI14 > 50 (strong momentum) AND price < EMA14 (pullback in downtrend) AND weekly downtrend AND volume
            elif rsi7[i] > 70 and rsi14[i] > 50 and close[i] < ema14[i] and not weekly_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI7 > 50 (momentum recovered) OR weekly trend turns down
            if rsi7[i] > 50 or not weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI7 < 50 (momentum weakened) OR weekly trend turns up
            if rsi7[i] < 50 or weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals