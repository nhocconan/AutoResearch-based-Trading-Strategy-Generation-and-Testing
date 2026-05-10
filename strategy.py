#!/usr/bin/env python3
# 1d_KeltnerChannel_Breakout_With_VolumeConfirmation
# Hypothesis: Long when price closes above upper Keltner Channel (EMA20, ATRx2) with volume > 1.3x average in uptrend (price > EMA50).
# Short when price closes below lower Keltner Channel (EMA20, ATRx2) with volume > 1.3x average in downtrend (price < EMA50).
# Exit when price re-enters Keltner Channel or ATR-based stoploss hit.
# Uses volatility-based breakouts to capture trends in both bull and bear markets.
# Designed for 15-25 trades/year on daily timeframe to minimize fee drag.

name = "1d_KeltnerChannel_Breakout_With_VolumeConfirmation"
timeframe = "1d"
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
    
    # Calculate ATR(20) for Keltner Channel and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # EMA20 for Keltner Channel center
    ema_20 = np.full(n, np.nan)
    for i in range(20, n):
        if i == 20:
            ema_20[i] = np.mean(close[:20])
        else:
            ema_20[i] = (close[i] * 2/21) + (ema_20[i-1] * 19/21)
    
    # Keltner Channel bands (EMA20 ± ATRx2)
    upper_kc = ema_20 + 2 * atr
    lower_kc = ema_20 - 2 * atr
    
    # EMA50 for trend filter
    ema_50 = np.full(n, np.nan)
    for i in range(50, n):
        if i == 50:
            ema_50[i] = np.mean(close[:50])
        else:
            ema_50[i] = (close[i] * 2/51) + (ema_50[i-1] * 49/51)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(upper_kc[i]) or np.isnan(lower_kc[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of EMA50 trend
            if close[i] > ema_50[i]:  # Uptrend
                # Long: Close above upper KC with volume confirmation
                if close[i] > upper_kc[i] and volume[i] > 1.3 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Close below lower KC with volume confirmation
                if close[i] < lower_kc[i] and volume[i] > 1.3 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price re-enters Keltner Channel or stoploss hit
            if close[i] < upper_kc[i] or (i > 0 and low[i] < ema_50[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters Keltner Channel or stoploss hit
            if close[i] > lower_kc[i] or (i > 0 and high[i] > ema_50[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals