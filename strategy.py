#!/usr/bin/env python3
"""
4h_SwingBreakout_VolumeRegime
Hypothesis: Swing high/low breakouts with volume confirmation and regime filter (ADX) capture trends while avoiding whipsaws. 
Works in bull/bear by requiring trend alignment (ADX > 25) and volume surge. 
Target: 100-200 trades over 4 years (25-50/year) with discrete sizing (0.30).
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
    
    # Swing points: 5-bar lookback/forward (confirmed after 2nd bar)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    for i in range(2, n-2):
        if high[i] == np.max(high[i-2:i+3]) and high[i] >= np.max(high[i-2:i+3]):
            swing_high[i] = high[i]
        if low[i] == np.min(low[i-2:i+3]) and low[i] <= np.min(low[i-2:i+3]):
            swing_low[i] = low[i]
    
    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # Regime filter: ADX > 25 = trending (favor breakouts)
    regime_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.30   # Position size: 30% of capital
    
    # Warmup: need swing points (2), ADX (14+14), volume avg (20)
    start_idx = max(2, 14+14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(swing_high[i]) and np.isnan(swing_low[i])) or \
           np.isnan(adx[i]) or np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        sh = swing_high[i]
        sl = swing_low[i]
        adx_val = adx[i]
        vol_conf = volume_confirm[i]
        regime_ok = regime_filter[i]
        
        if position == 0:
            # Long when price breaks above recent swing high with volume and trending regime
            if not np.isnan(sh) and close_val > sh and vol_conf and regime_ok:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short when price breaks below recent swing low with volume and trending regime
            elif not np.isnan(sl) and close_val < sl and vol_conf and regime_ok:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: close below swing low or ATR-based stop (2*ATR)
            if not np.isnan(sl) and close_val < sl:
                signals[i] = 0.0
                position = 0
            else:
                atr_val = atr[i]
                stop_loss = entry_price - 2.0 * atr_val
                if close_val <= stop_loss:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Exit: close above swing high or ATR-based stop (2*ATR)
            if not np.isnan(sh) and close_val > sh:
                signals[i] = 0.0
                position = 0
            else:
                atr_val = atr[i]
                stop_loss = entry_price + 2.0 * atr_val
                if close_val >= stop_loss:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "4h_SwingBreakout_VolumeRegime"
timeframe = "4h"
leverage = 1.0