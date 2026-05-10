#!/usr/bin/env python3
# 4h_AdaptiveTrend_With_VolumeConfirmation
# Hypothesis: Long when price closes above upper Bollinger Band (20,2) with volume > 1.5x average in uptrend (price > EMA50).
# Short when price closes below lower Bollinger Band (20,2) with volume > 1.5x average in downtrend (price < EMA50).
# Exit when price re-enters Bollinger Bands or ATR-based stoploss hit.
# Uses Bollinger Bands for volatility-based breakouts, works in both bull and bear markets by following the trend.
# Designed for 20-50 trades/year to avoid fee drag.

name = "4h_AdaptiveTrend_With_VolumeConfirmation"
timeframe = "4h"
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    bb_std_dev = np.full(n, np.nan)
    upper_bb = np.full(n, np.nan)
    lower_bb = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        bb_std_dev[i] = np.std(close[i-bb_period:i])
        upper_bb[i] = sma[i] + bb_std * bb_std_dev[i]
        lower_bb[i] = sma[i] - bb_std * bb_std_dev[i]
    
    # Get EMA50 for trend filter
    ema_50 = np.full(n, np.nan)
    for i in range(50, n):
        ema_50[i] = np.mean(close[i-50:i])  # Simple MA for efficiency
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of EMA50 trend
            if close[i] > ema_50[i]:  # Uptrend
                # Long: Close above upper BB with volume confirmation
                if close[i] > upper_bb[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Close below lower BB with volume confirmation
                if close[i] < lower_bb[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price re-enters Bollinger Bands or stoploss hit
            if close[i] < upper_bb[i] or (i > 0 and low[i] < ema_50[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters Bollinger Bands or stoploss hit
            if close[i] > lower_bb[i] or (i > 0 and high[i] > ema_50[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals