# 6h_RSI_Divergence_4HTrend_1DVolume
# Hypothesis: Detects momentum exhaustion via RSI divergence on 6h chart, with 4h trend filter and 1d volume confirmation.
# RSI divergence signals potential reversals in both bull and bear markets. The 4h trend filter ensures trades align with intermediate-term momentum,
# while 1d volume surge confirms institutional participation. Target: 15-25 trades/year per symbol to minimize fee drag.

timeframe = "6h"
name = "6h_RSI_Divergence_4HTrend_1DVolume"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 6h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate 20-period EMA on 4h close for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 20-period average volume on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 15  # Need at least 14+1 for RSI
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or np.isnan(rsi[i-2]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for RSI divergence over last 3 bars
        bullish_div = False
        bearish_div = False
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if low[i] < low[i-1] < low[i-2] and rsi[i] > rsi[i-1] > rsi[i-2]:
            bullish_div = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        elif high[i] > high[i-1] > high[i-2] and rsi[i] < rsi[i-1] < rsi[i-2]:
            bearish_div = True
        
        if position == 0:
            # Long: bullish RSI divergence + price above 4h EMA20 (uptrend) + volume surge
            if bullish_div and close[i] > ema_20_4h_aligned[i] and volume[i] > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence + price below 4h EMA20 (downtrend) + volume surge
            elif bearish_div and close[i] < ema_20_4h_aligned[i] and volume[i] > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish RSI divergence or price breaks below 4h EMA20
            if bearish_div or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish RSI divergence or price breaks above 4h EMA20
            if bullish_div or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals