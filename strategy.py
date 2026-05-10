#!/usr/bin/env python3
# 1d_RSI_MeanReversion_1wTrend_Filter
# Hypothesis: Daily RSI oversold/overbought with weekly trend filter provides mean reversion entries in ranging markets and trend continuation in strong trends.
# Weekly EMA34 determines trend direction: only take long signals when price above weekly EMA34, short when below.
# RSI(14) < 30 for long, > 70 for short. Volume confirmation requires current volume > 1.5x 20-day EMA volume.
# Designed to work in both bull (trend continuation) and bear (mean reversion during pullbacks) markets.
# Target: 10-25 trades/year on 1d timeframe.

name = "1d_RSI_MeanReversion_1wTrend_Filter"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter: current volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (34) + daily RSI (14) + volume EMA (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND price above weekly EMA34 AND volume confirmation
            if rsi[i] < 30 and close[i] > ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) AND price below weekly EMA34 AND volume confirmation
            elif rsi[i] > 70 and close[i] < ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) OR price crosses below weekly EMA34
            if rsi[i] > 50 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) OR price crosses above weekly EMA34
            if rsi[i] < 50 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals