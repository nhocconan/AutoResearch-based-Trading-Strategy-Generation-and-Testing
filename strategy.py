#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND Bull Power > 0 AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when Bull Power and Bear Power converge (both cross zero) or opposite power becomes dominant.
Uses 1w HTF for primary trend direction (avoids counter-trend trades in strong bull/bear markets). Target: 50-150 total trades over 4 years (12-37/year).
Elder Ray measures bull/bear strength via EMA13; combining with weekly trend filter captures strong momentum moves with proper directional bias.
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray components (Bull Power and Bear Power)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # EMA13 (13), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50_1w_val = ema_50_1w_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND close > 1w EMA50 AND volume spike
            if bull_val > 0 and bear_val < 0 and price > ema50_1w_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND close < 1w EMA50 AND volume spike
            elif bear_val < 0 and bull_val > 0 and price < ema50_1w_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when power converges (both cross zero) or opposite power becomes dominant
            if position == 1:
                # Exit long when Bull Power <= 0 OR Bear Power >= 0 (convergence or bearish takeover)
                if bull_val <= 0 or bear_val >= 0:
                    exit_signal = True
                else:
                    exit_signal = False
            else:  # position == -1
                # Exit short when Bear Power >= 0 OR Bull Power <= 0 (convergence or bullish takeover)
                if bear_val >= 0 or bull_val <= 0:
                    exit_signal = True
                else:
                    exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1wEMA50_Trend_VolumeConfirmation_PowerConvergenceExit"
timeframe = "6h"
leverage = 1.0