#!/usr/bin/env python3
name = "4h_RSI_Div_Volume_1dTrend"
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
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume MA20 for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure RSI and other indicators are stable
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or np.isnan(rsi[i-2]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish RSI divergence: price makes lower low, RSI makes higher low
        bullish_div = (low[i] < low[i-1] and low[i-1] < low[i-2]) and \
                      (rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2])
        # Bearish RSI divergence: price makes higher high, RSI makes lower high
        bearish_div = (high[i] > high[i-1] and high[i-1] > high[i-2]) and \
                      (rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2])
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: bullish RSI divergence + price above 1d EMA200 + volume confirmation
            if bullish_div and close[i] > ema_200_1d_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence + price below 1d EMA200 + volume confirmation
            elif bearish_div and close[i] < ema_200_1d_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish RSI divergence or price below 1d EMA200
            if bearish_div or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish RSI divergence or price above 1d EMA200
            if bullish_div or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h RSI divergence with 1d trend filter and volume confirmation
# - Bullish divergence: price makes lower low while RSI makes higher low (momentum weakening)
# - Bearish divergence: price makes higher high while RSI makes lower high (momentum weakening)
# - Works in both bull and bear markets: divergences signal reversals regardless of trend
# - 1d EMA200 filter ensures trades align with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Exits on opposite divergence or trend violation to capture mean reversion
# - Position size 0.25 limits risk and keeps trade frequency ~20-50/year
# - RSI divergence is a robust reversal signal effective in ranging and trending markets
# - Aims for 80-150 total trades over 4 years (20-38/year) to stay within limits
# - Combines classical momentum divergence with trend filtering for robustness
# - Avoids overtrading by requiring multiple confluence factors for entry