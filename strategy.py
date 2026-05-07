#!/usr/bin/env python3
name = "4h_KAMA_Trend_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA trend filter
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_trend = close_1d > kama_1d  # True for uptrend
    kama_1d_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_trend.astype(float))
    
    # Calculate 4h RSI for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~8 hours for 4h
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_trend_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction from 1d KAMA
        trend_up = kama_1d_trend_aligned[i] > 0.5
        trend_down = kama_1d_trend_aligned[i] < 0.5
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: RSI oversold in uptrend with volume confirmation
            if (rsi[i] < 30 and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: RSI overbought in downtrend with volume confirmation
            elif (rsi[i] > 70 and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: RSI returns to neutral or trend changes
            if (rsi[i] > 50 or not trend_up):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or trend changes
            if (rsi[i] < 50 or not trend_down):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combining 1d KAMA trend filter with 4h RSI mean reversion and volume confirmation
# will capture trend-aligned reversals in both bull and bear markets. KAMA adapts to market
# conditions, reducing whipsaws. RSI <30/>70 provides clear entry signals, while volume
# confirmation ensures institutional participation. Position size 0.25 limits drawdown.
# Expected trades: 20-40 per year, avoiding overtrading. Works on BTC/ETH as primary targets.