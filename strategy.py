#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v4
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1-day EMA34 trend filter, volume spike (>2x 20-bar avg), and ATR-based stoploss capture institutional breakouts while minimizing whipsaws. Uses discrete position sizing (0.25) to reduce fee churn. Targets 20-50 trades/year to stay within fee drag limits while maintaining edge through trend and volume confirmation. Works in both bull (breakouts with momentum) and bear (mean reversion at extremes) markets.
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (1-bar delay for completed daily bar)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume average (20-period = ~3.3h on 4h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_bar = 0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 34, 14, 2)  # volume MA, daily EMA, ATR, 1d data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_1d_val = r1_1d_aligned[i]
        s1_1d_val = s1_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        atr_val = atr[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2x 20-period average (tighter)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above R1 with uptrend (close > EMA34) and volume confirmation
            long_signal = (high_val > r1_1d_val) and (close_val > ema_34_1d_val) and volume_confirmed
            # Short: price breaks below S1 with downtrend (close < EMA34) and volume confirmation
            short_signal = (low_val < s1_1d_val) and (close_val < ema_34_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                entry_bar = i
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                entry_bar = i
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. ATR-based stoploss: 2.0 * ATR below entry
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 2. Opposite breakout: price breaks below S1 (exit long)
            elif low_val < s1_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 3. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 4. Time-based exit: max 10 bars (~40h) to prevent stagnation
            elif i - entry_bar >= 10:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. ATR-based stoploss: 2.0 * ATR above entry
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 2. Opposite breakout: price breaks above R1 (exit short)
            elif high_val > r1_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 3. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 4. Time-based exit: max 10 bars (~40h) to prevent stagnation
            elif i - entry_bar >= 10:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v4"
timeframe = "4h"
leverage = 1.0