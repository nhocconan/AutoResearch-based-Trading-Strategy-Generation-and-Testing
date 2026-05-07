#!/usr/bin/env python3
name = "4h_RSI_20_80_With_Trend_and_Volume"
timeframe = "4h"
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
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    trend_up = close > ema50_1d_aligned
    trend_down = close < ema50_1d_aligned
    
    # RSI(14) - momentum oscillator
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<20) with volume surge and 1d uptrend
            if rsi[i] < 20 and vol_surge[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>80) with volume surge and 1d downtrend
            elif rsi[i] > 80 and vol_surge[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought (>70) or trend turns down
            if rsi[i] > 70 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold (<30) or trend turns up
            if rsi[i] < 30 or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI extreme readings (20/80) with volume confirmation and 1d trend filter capture mean-reversion bounces in trending markets.
# Long when RSI < 20 (oversold) with volume surge in 1d uptrend - buys pullbacks in uptrends.
# Short when RSI > 80 (overbought) with volume surge in 1d downtrend - sells rallies in downtrends.
# Uses 1d EMA50 for trend filter to ensure we trade with the higher timeframe trend.
# Volume surge confirms conviction behind the move.
# Designed for 4h timeframe to balance trade frequency (~20-50/year) and avoid whipsaws.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).