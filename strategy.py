#!/usr/bin/env python3
"""
6h_ElderRay_Pullback_1dTrend_Volume
Hypothesis: Elder Ray (Bull/Bear Power) pullback trades in direction of 1d EMA34 trend with volume confirmation.
Works in bull/bear by following daily trend. Long when Bull Power > 0 and price pulls back to EMA13 in uptrend.
Short when Bear Power < 0 and price pulls back to EMA13 in downtrend. Volume spike confirms institutional interest.
Target: 12-30 trades/year.
"""

name = "6h_ElderRay_Pullback_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # EMA13 for pullback entry (on 6h timeframe)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike: current volume > 1.8x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 13)  # volume SMA, EMA34, EMA13 warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 0:
            # Long: Uptrend (price > daily EMA34), Bull Power > 0, pullback to EMA13
            if close[i] > ema_34_1d_aligned[i] and bull_power[i] > 0 and low[i] <= ema_13[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < daily EMA34), Bear Power < 0, pullback to EMA13
            elif close[i] < ema_34_1d_aligned[i] and bear_power[i] < 0 and high[i] >= ema_13[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Trend reversal or Bear Power negative
            if close[i] < ema_34_1d_aligned[i] or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend reversal or Bull Power positive
            if close[i] > ema_34_1d_aligned[i] or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals