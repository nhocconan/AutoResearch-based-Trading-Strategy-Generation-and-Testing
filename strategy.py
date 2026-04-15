#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA trend and 1d RSI momentum with volume confirmation
# Uses 4h EMA200 for trend direction (bull/bear filter) and 1d RSI(14) for momentum strength
# Volume filter ensures trades occur during high conviction periods
# Designed for low trade frequency (target 15-35/year) to avoid fee drag
# Works in bull markets (trend + momentum) and bear markets (counter-trend bounces when RSI extremes)
# Uses discrete position sizing (0.20) to minimize churn

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data once
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 1d RSI(14) for momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # 1h volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Get aligned indicators
        ema200_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)[i]
        rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        
        # Skip if not enough data
        if np.isnan(ema200_aligned) or np.isnan(rsi_aligned) or np.isnan(volume_ma[i]):
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * volume_ma[i]
        
        # Long conditions: price above 4h EMA200 (bullish trend) AND RSI > 50 (bullish momentum)
        if close[i] > ema200_aligned and rsi_aligned > 50 and vol_confirm and position <= 0:
            position = 1
            signals[i] = position_size
        # Short conditions: price below 4h EMA200 (bearish trend) AND RSI < 50 (bearish momentum)
        elif close[i] < ema200_aligned and rsi_aligned < 50 and vol_confirm and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit: trend reversal (price crosses EMA200) or momentum dies (RSI crosses 50 opposite)
        elif position == 1 and (close[i] < ema200_aligned or rsi_aligned < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema200_aligned or rsi_aligned > 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4h_EMA200_1d_RSI_Volume"
timeframe = "1h"
leverage = 1.0