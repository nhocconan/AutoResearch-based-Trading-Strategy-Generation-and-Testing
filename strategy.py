# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h RSI Extreme + Volume Spike + Trend Filter
Long: RSI < 30 (oversold) + volume > 2x 20-period volume SMA + close > 200-bar EMA
Short: RSI > 70 (overbought) + volume > 2x 20-period volume SMA + close < 200-bar EMA
Exit: RSI crosses back to 50 (mean reversion completion)
Designed to capture mean reversion moves in both bull and bear markets with volume confirmation.
Target: 20-50 total trades over 4 years (5-12/year) - well within limits to avoid fee drag.
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
    
    # Calculate RSI(14) on price
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate 200-bar EMA for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume SMA(20)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 200  # need EMA200
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_200[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        rsi_val = rsi[i]
        ema_val = ema_200[i]
        
        if position == 0:
            # Long: RSI oversold + volume spike + above EMA200
            if rsi_val < 30 and vol > 2.0 * vol_sma_val and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + volume spike + below EMA200
            elif rsi_val > 70 and vol > 2.0 * vol_sma_val and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses back to 50 (mean reversion complete)
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses back to 50 (mean reversion complete)
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Extreme_VolumeSpike_EMA200"
timeframe = "4h"
leverage = 1.0