#!/usr/bin/env python3
"""
4h_Parabolic_SAR_12hTrend_Volume
Parabolic SAR breakout with 12-hour trend filter and volume confirmation.
In bull markets (price > 12h EMA50), long on SAR flip to below price with volume.
In bear markets (price < 12h EMA50), short on SAR flip to above price with volume.
Parabolic SAR adapts to volatility, reducing false signals in ranging markets.
Target: 20-35 trades per year (~80-140 over 4 years) with position size 0.25.
"""

name = "4h_Parabolic_SAR_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12-hour EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Parabolic SAR calculation
    # Start with assumption of long position
    psar = np.zeros(n)
    psar[0] = low[0]  # Start SAR at first low
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02  # Acceleration factor
    max_af = 0.2  # Maximum acceleration factor
    ep = high[0]  # Extreme point
    
    for i in range(1, n):
        if trend == 1:  # Uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't exceed previous two lows
            psar[i] = min(psar[i], low[i-1])
            if i >= 2:
                psar[i] = min(psar[i], low[i-2])
            
            # Trend reversal check
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                # Continue uptrend
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:  # Downtrend
            psar[i] = psar[i-1] + af * (psar[i-1] - ep)
            # Ensure SAR doesn't go below previous two highs
            psar[i] = max(psar[i], high[i-1])
            if i >= 2:
                psar[i] = max(psar[i], high[i-2])
            
            # Trend reversal check
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                # Continue downtrend
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(psar[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 12-hour EMA50
        uptrend_regime = close[i] > ema_50_12h_aligned[i]
        downtrend_regime = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: SAR flips below price (bullish) in uptrend regime + volume
            long_entry = (psar[i] < close[i]) and (psar[i-1] >= close[i-1]) and uptrend_regime and volume_confirm
            # Short: SAR flips above price (bearish) in downtrend regime + volume
            short_entry = (psar[i] > close[i]) and (psar[i-1] <= close[i-1]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: SAR flips above price (bearish) or regime changes to downtrend
            if (psar[i] > close[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: SAR flips below price (bullish) or regime changes to uptrend
            if (psar[i] < close[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals