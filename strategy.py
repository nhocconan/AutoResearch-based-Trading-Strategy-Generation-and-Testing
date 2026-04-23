#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA Trend + Donchian Breakout with Volume Spike Filter and ATR Trailing Stop.
Long when KAMA rising, price breaks above Donchian(20) high, and volume > 2x 20-period average.
Short when KAMA falling, price breaks below Donchian(20) low, and volume > 2x 20-period average.
Exit via ATR-based trailing stop (3*ATR) or opposing signal.
Uses discrete position sizing (0.30) and volume confirmation to reduce false breakouts.
Designed for 4h timeframe to target 20-50 trades/year per symbol with Sharpe > 0 on BTC/ETH in both bull and bear markets.
"""

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
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0])).rolling(window=10, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for trailing stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0
    lowest_since_short = 0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 14, 20)  # Ensure warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, price breaks above Donchian high, volume spike
            if (kama[i] > kama[i-1] and 
                close[i] > donch_high[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.30
                position = 1
                highest_since_long = close[i]
            # Short: KAMA falling, price breaks below Donchian low, volume spike
            elif (kama[i] < kama[i-1] and 
                  close[i] < donch_low[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.30
                position = -1
                lowest_since_short = close[i]
        else:
            # Update highest/lowest for trailing stop
            if position == 1:
                highest_since_long = max(highest_since_long, close[i])
                # Exit if price drops 3*ATR from highest or KAMA turns down
                if (close[i] < highest_since_long - 3.0 * atr[i] or 
                    kama[i] < kama[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                lowest_since_short = min(lowest_since_short, close[i])
                # Exit if price rises 3*ATR from lowest or KAMA turns up
                if (close[i] > lowest_since_short + 3.0 * atr[i] or 
                    kama[i] > kama[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4H_KAMA_Donchian_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0