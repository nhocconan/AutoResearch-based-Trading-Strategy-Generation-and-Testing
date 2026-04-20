#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI divergence with 1d trend filter and volume confirmation
# - Long when: 1d EMA200 up + 4h RSI < 30 (oversold) + volume > 1.5x 20-period average
# - Short when: 1d EMA200 down + 4h RSI > 70 (overbought) + volume > 1.5x 20-period average
# - Exit when: RSI returns to neutral zone (40-60) or opposite extreme reached
# - Uses 1d for trend filter (EMA200) and 4h for RSI and execution
# - Designed to work in both bull (buy dips) and bear (sell rallies) markets
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_slope = np.diff(ema200_1d, prepend=ema200_1d[0])
    ema200_up = ema200_1d_slope > 0
    ema200_down = ema200_1d_slope < 0
    
    # Align trend filter to 4h
    ema200_up_4h = align_htf_to_ltf(prices, df_1d, ema200_up)
    ema200_down_4h = align_htf_to_ltf(prices, df_1d, ema200_down)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema200_up_4h[i]) or np.isnan(ema200_down_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: uptrend + RSI oversold + volume surge
            if ema200_up_4h[i] and rsi[i] < 30 and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + RSI overbought + volume surge
            elif ema200_down_4h[i] and rsi[i] > 70 and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or becomes overbought
            if rsi[i] >= 40:  # Exit when RSI reaches neutral or overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or becomes oversold
            if rsi[i] <= 60:  # Exit when RSI reaches neutral or oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Divergence_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0