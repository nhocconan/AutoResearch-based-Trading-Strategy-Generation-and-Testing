# 102302
#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Divergence
Hypothesis: KAMA identifies trend direction; RSI divergence on momentum exhaustion provides high-probability reversal entries.
Works in bull markets (buy bullish RSI divergence in uptrend) and bear markets (sell bearish RSI divergence in downtrend).
Target: 20-30 trades/year via strict divergence requirements + KAMA trend filter + volume confirmation.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend filter (ER=10)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder - will compute properly
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i >= 10:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility_sum if volatility_sum > 0 else 0
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI for divergence (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d KAMA
        uptrend = close[i] > kama_1d_aligned[i]
        downtrend = close[i] < kama_1d_aligned[i]
        
        # RSI divergence detection (look back 5 bars for swing)
        bullish_div = False
        bearish_div = False
        if i >= 5:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if low[i] < low[i-5] and rsi[i] > rsi[i-5]:
                # Confirm with higher low in price
                if low[i] > low[i-1] and low[i-1] > low[i-2]:
                    bullish_div = True
            # Bearish divergence: price makes higher high, RSI makes lower high
            if high[i] > high[i-5] and rsi[i] < rsi[i-5]:
                # Confirm with lower high in price
                if high[i] < high[i-1] and high[i-1] < high[i-2]:
                    bearish_div = True
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic
        long_entry = vol_confirm and uptrend and bullish_div
        short_entry = vol_confirm and downtrend and bearish_div
        
        # Exit logic: opposite divergence or trend change
        long_exit = bearish_div or (not uptrend)
        short_exit = bullish_div or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Divergence"
timeframe = "4h"
leverage = 1.0