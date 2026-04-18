#!/usr/bin/env python3
"""
4h_RSI_Momentum_With_Trend
Hypothesis: RSI momentum (cross above 50) in the direction of the 4h EMA trend, with volume confirmation, yields fewer false signals than pure RSI extremes.
Works in bull markets (trend-following) and bear markets (only trades when RSI>50 in downtrend for shorts).
Target: 20-40 trades/year to minimize fee drag while capturing momentum with trend alignment.
"""

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
    
    # Trend: 4h EMA34
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Warmup for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_val = ema_34[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: RSI crosses above 50, price above EMA, volume spike
            if rsi_val > 50 and rsi[i-1] <= 50 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50, price below EMA, volume spike
            elif rsi_val < 50 and rsi[i-1] >= 50 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI falls below 40 OR price crosses below EMA
            if rsi_val < 40 or price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI rises above 60 OR price crosses above EMA
            if rsi_val > 60 or price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_RSI_Momentum_With_Trend"
timeframe = "4h"
leverage = 1.0