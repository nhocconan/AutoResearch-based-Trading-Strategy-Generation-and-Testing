# 4h_12h_1d_Combined_Momentum
# Combines 12h EMA trend, 1d RSI momentum, and 4h price action with volume confirmation
# Designed for low trade frequency (~30-50/year) to avoid fee drag while capturing trends in both bull and bear markets
# Uses discrete position sizing (0.25) to minimize churn

#!/usr/bin/env python3
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
    
    # Load 12h and 1d data once
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 1d RSI(14) for momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # 4h volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned indicators
        ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)[i]
        rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        
        # Skip if not enough data
        if np.isnan(ema50_aligned) or np.isnan(rsi_aligned) or np.isnan(volume_ma[i]):
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Long conditions: price above 12h EMA50 AND RSI > 50 (bullish momentum)
        if close[i] > ema50_aligned and rsi_aligned > 50 and vol_confirm and position <= 0:
            position = 1
            signals[i] = position_size
        # Short conditions: price below 12h EMA50 AND RSI < 50 (bearish momentum)
        elif close[i] < ema50_aligned and rsi_aligned < 50 and vol_confirm and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit: momentum reversal (RSI crosses 50) or volume dies
        elif position == 1 and (rsi_aligned < 50 or volume[i] < 0.5 * volume_ma[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned > 50 or volume[i] < 0.5 * volume_ma[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_12h_1d_Combined_Momentum"
timeframe = "4h"
leverage = 1.0