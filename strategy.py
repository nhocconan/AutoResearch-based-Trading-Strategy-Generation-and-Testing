#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy using 4h RSI divergence with volume confirmation
# Uses 4h RSI(14) for momentum direction and 1h volume spike for entry timing
# Enters long when 4h RSI < 40 (bullish momentum) and 1h volume > 2x 20-period average
# Enters short when 4h RSI > 60 (bearish momentum) and 1h volume > 2x 20-period average
# Designed for low trade frequency (target 15-30/year) to avoid fee drag
# Works in trending markets (momentum continuation) and avoids ranging markets
# Uses discrete position sizing (0.20) to minimize churn

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h RSI(14) for momentum
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # 1h volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            continue
        
        # Get aligned 4h RSI
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)[i]
        
        # Skip if not enough data
        if np.isnan(rsi_4h_aligned) or np.isnan(volume_ma[i]):
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * volume_ma[i]
        
        # Long conditions: 4h RSI < 40 (bullish momentum) AND volume spike
        if rsi_4h_aligned < 40 and vol_confirm and position <= 0:
            position = 1
            signals[i] = position_size
        # Short conditions: 4h RSI > 60 (bearish momentum) AND volume spike
        elif rsi_4h_aligned > 60 and vol_confirm and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit: 4h RSI returns to neutral zone (40-60)
        elif position == 1 and rsi_4h_aligned >= 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi_4h_aligned <= 50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4h_RSI_Momentum_Volume"
timeframe = "1h"
leverage = 1.0