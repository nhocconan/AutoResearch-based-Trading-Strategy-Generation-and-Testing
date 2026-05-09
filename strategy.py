#!/usr/bin/env python3
"""
1h_4h_1d_Trend_Following_with_Volume_Confirmation
Hypothesis: Use 4h trend (EMA34) and 1d momentum (RSI) for direction, with 1h entry on pullbacks to EMA21 confirmed by volume spikes. This captures trend continuation in both bull and bear markets while avoiding counter-trend trades. Low frequency expected due to multiple confluence requirements.
"""

name = "1h_4h_1d_Trend_Following_with_Volume_Confirmation"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 for trend direction
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d RSI for momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h EMA21 for entry timing
    ema_21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    # Volume spike: current volume / 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    volume_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 21, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_21[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h uptrend, 1d bullish momentum, price pulls back to EMA21 with volume spike
            if (ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and  # 4h EMA rising
                rsi_1d_aligned[i] > 50 and  # 1d bullish momentum
                close[i] <= ema_21[i] * 1.005 and  # Near or slightly above EMA21 (pullback)
                volume_ratio[i] > 1.5):  # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend, 1d bearish momentum, price bounces to EMA21 with volume spike
            elif (ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and  # 4h EMA falling
                  rsi_1d_aligned[i] < 50 and  # 1d bearish momentum
                  close[i] >= ema_21[i] * 0.995 and  # Near or slightly below EMA21 (bounce)
                  volume_ratio[i] > 1.5):  # Volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend turns down OR 1d momentum turns bearish
            if (ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] or  # 4h EMA falling
                rsi_1d_aligned[i] < 50):  # 1d momentum bearish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend turns up OR 1d momentum turns bullish
            if (ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] or  # 4h EMA rising
                rsi_1d_aligned[i] > 50):  # 1d momentum bullish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals