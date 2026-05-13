#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend with 1w EMA34 filter and volume confirmation.
# Long when KAMA direction is up AND 1w EMA34 is rising AND volume > 1.8x 20-period average.
# Short when KAMA direction is down AND 1w EMA34 is falling AND volume > 1.8x 20-period average.
# Uses ATR(14) trailing stop (2.0x) for risk control.
# Discrete position sizing (0.25) to minimize fee churn.
# Target: 50-100 total trades over 4 years (12-25/year) on 1d.

name = "1d_KAMA_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    def kama(source, period=10, fast=2, slow=30):
        """Kaufman Adaptive Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=np.float64)
        
        # Efficiency Ratio
        change = np.abs(np.diff(source, n=period))
        volatility = np.sum(np.abs(np.diff(source)), axis=0) if len(source) > 1 else 0
        er = np.zeros_like(source)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        result = np.full_like(source, np.nan, dtype=np.float64)
        result[period-1] = np.mean(source[:period])
        for i in range(period, len(source)):
            result[i] = result[i-1] + sc[i] * (source[i] - result[i-1])
        return result
    
    kama_values = kama(close, 10, 2, 30)
    kama_up = kama_values > np.roll(kama_values, 1)
    kama_down = kama_values < np.roll(kama_values, 1)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe (wait for 1w bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama_values[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA up AND 1w EMA34 rising AND volume spike
            if (kama_up[i] and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: KAMA down AND 1w EMA34 falling AND volume spike
            elif (kama_down[i] and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals