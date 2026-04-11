#!/usr/bin/env python3
"""
12h_1d_34ema_rsi_volume_v1
Strategy: 12h EMA34 + RSI(14) + Volume Spike
Timeframe: 12h
Leverage: 1.0
Hypothesis: Combines 12h EMA34 trend filter with RSI mean reversion and volume confirmation. Enters long when price pulls back to EMA34 with RSI < 40 and volume spike; enters short when price rallies to EMA34 with RSI > 60 and volume spike. Designed to capture mean reversion within the trend, avoiding overextended moves. Uses volume to filter false signals. Target: 60-120 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_34ema_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d RSI(14) for mean reversion signals
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Mean reversion signals with volume confirmation
        long_signal = (price_close <= ema_34[i]) and (rsi_1d_aligned[i] < 40) and vol_spike[i]
        short_signal = (price_close >= ema_34[i]) and (rsi_1d_aligned[i] > 60) and vol_spike[i]
        
        # Exit when price crosses EMA34 in opposite direction
        exit_long = position == 1 and price_close > ema_34[i]
        exit_short = position == -1 and price_close < ema_34[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals