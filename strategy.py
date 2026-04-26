#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA). Exits via ATR(14) trailing stop (2.0 ATR from extreme) or opposite breakout. Designed for lower frequency (target 20-50 trades/year) to avoid fee drag, works in bull/bear via trend alignment. Uses discrete position sizing (0.25) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR(14) for stoploss calculation
    atr_period = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Track extreme for trailing stop
    long_high = 0.0
    low_low = 0.0
    
    # Warmup: max of calculations
    start_idx = max(20, 34, 20, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Previous day's Camarilla levels (R1, S1)
        if i >= 24:  # Need previous day's data (24*4h bars in 1d)
            prev_high = np.max(high[i-24:i])
            prev_low = np.min(low[i-24:i])
            prev_close = close[i-1]  # Previous bar's close
            pivot = (prev_high + prev_low + prev_close) / 3.0
            r1 = pivot + (1.1/12.0) * (prev_high - prev_low)
            s1 = pivot - (1.1/12.0) * (prev_high - prev_low)
        else:
            # Not enough data for Camarilla calculation
            r1 = high_val
            s1 = low_val
        
        # Entry conditions
        long_entry = (close_val > r1) and bullish_1d and vol_spike
        short_entry = (close_val < s1) and bearish_1d and vol_spike
        
        # Update trailing extremes
        if position == 1:
            long_high = max(long_high, high_val)
        elif position == -1:
            low_low = min(low_low, low_val)
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # ATR trailing stop or opposite breakout
            if long_high > 0 and close_val < (long_high - 2.0 * atr_val):
                exit_long = True
            elif close_val < s1:  # Opposite breakout (S1 level)
                exit_long = True
        elif position == -1:
            # ATR trailing stop or opposite breakout
            if low_low > 0 and close_val > (low_low + 2.0 * atr_val):
                exit_short = True
            elif close_val > r1:  # Opposite breakout (R1 level)
                exit_short = True
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            long_high = high_val  # Reset extreme on new entry
            low_low = 0.0
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            low_low = low_val  # Reset extreme on new entry
            long_high = 0.0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            long_high = 0.0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            low_low = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0