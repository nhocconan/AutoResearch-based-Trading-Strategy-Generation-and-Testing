#!/usr/bin/env python3
"""
1d_KAMA_Direction_1wTrend_VolumeSpike_v1
Hypothesis: Daily KAMA trend direction with weekly EMA34 filter and volume spike confirmation.
- Long when KAMA(10,2,30) is rising AND weekly EMA34 uptrend AND volume > 2.0 * volume_ma(20)
- Short when KAMA(10,2,30) is falling AND weekly EMA34 downtrend AND volume > 2.0 * volume_ma(20)
- KAMA adapts to market noise, reducing whipsaws in ranging markets
- Weekly EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume spike (2.0x) confirms institutional participation and reduces false signals
- Designed for low frequency (target 10-25 trades/year on 1d) to minimize fee drag
- Novelty: Combines adaptive KAMA trend with weekly trend filter for robust performance in bull/bear
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter (needs completed weekly candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1w = np.where(ema_34_1w_aligned > 0, 
                        np.where(close > ema_34_1w_aligned, 1, -1), 
                        0)
    
    # Calculate KAMA(10,2,30) - ER=10, fastest=2, slowest=30
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fastest = 2.0 / (2 + 1)  # EMA(2)
    slowest = 2.0 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start calculation after 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 = rising, -1 = falling, 0 = flat
    kama_dir = np.where(kama > np.roll(kama, 1), 1, np.where(kama < np.roll(kama, 1), -1, 0))
    kama_dir[0] = 0  # First value has no previous
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 20 for volume MA, 30 for KAMA)
    start_idx = max(34, 20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_dir[i]) or np.isnan(trend_1w[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA direction + weekly trend + volume spike conditions
        if position == 0:
            # Long: KAMA rising AND weekly uptrend AND volume spike
            if kama_dir[i] == 1 and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND weekly downtrend AND volume spike
            elif kama_dir[i] == -1 and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA turns down OR weekly trend turns down
            if kama_dir[i] == -1 or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA turns up OR weekly trend turns up
            if kama_dir[i] == 1 or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0