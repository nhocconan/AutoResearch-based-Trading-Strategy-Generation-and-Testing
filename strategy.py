#!/usr/bin/env python3
"""
12h_PowerTrend_Composite_v1
12h strategy combining price action with momentum and volume confirmation.
- Long: Close > EMA(34) AND RSI(14) > 50 AND Volume > 1.5x avg volume
- Short: Close < EMA(34) AND RSI(14) < 50 AND Volume > 1.5x avg volume
- Exit: Opposite signal
Designed for 15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (momentum continuation) and bear markets (mean reversion via RSI extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA34 for trend
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema_34[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Momentum condition
        bullish_momentum = close[i] > ema_34[i] and rsi[i] > 50
        bearish_momentum = close[i] < ema_34[i] and rsi[i] < 50
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: bullish momentum + volume
            if bullish_momentum and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + volume
            elif bearish_momentum and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish momentum or volume confirmation
            if bearish_momentum or vol_confirm:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish momentum or volume confirmation
            if bullish_momentum or vol_confirm:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PowerTrend_Composite_v1"
timeframe = "12h"
leverage = 1.0