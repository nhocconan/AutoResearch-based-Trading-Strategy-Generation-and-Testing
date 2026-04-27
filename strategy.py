#!/usr/bin/env python3
"""
1d_KAMA_Trend_WeeklyEMA34_VolumeSpike
Hypothesis: Daily KAMA trend with weekly EMA34 filter and volume spike confirmation.
Long when KAMA trending up AND price > weekly EMA34 AND volume spike.
Short when KAMA trending down AND price < weekly EMA34 AND volume spike.
Exit on opposite KAMA direction or loss of weekly EMA34 alignment.
Designed for 10-25 trades/year on 1d to minimize fee drag while capturing strong multi-day moves aligned with weekly trend.
KAMA adapts to market noise, reducing whipsaws in choppy conditions.
Works in bull markets (adaptive uptrend with weekly support) and bear markets (adaptive downtrend with weekly resistance).
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
    
    # KAMA trend efficiency ratio (10-period)
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smooth ER with min_periods
    er_series = pd.Series(er, index=prices.index[-len(er):])
    er_smoothed = er_series.rolling(window=10, min_periods=1).mean().values
    # Fill beginning with zeros for alignment
    er_full = np.concatenate([np.full(10, np.nan), er_smoothed])
    
    # Fast and slow SC
    sc_fast = 2 / (2 + 1)
    sc_slow = 2 / (30 + 1)
    sc = er_full * (sc_fast - sc_slow) + sc_slow
    sc = np.where(np.isnan(sc), 0, sc)
    sc = sc * sc  # Square for smoothing
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(close)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, -1)
    
    # Weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for KAMA (10+), weekly EMA34 (~34*7=238 1d bars), volume avg
    start_idx = max(30, 238, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(kama_dir[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        kama_direction = kama_dir[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: KAMA direction with weekly EMA34 alignment and volume spike
            # Long: KAMA rising AND price > weekly EMA34 AND volume spike
            # Short: KAMA falling AND price < weekly EMA34 AND volume spike
            long_condition = (kama_direction == 1 and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (kama_direction == -1 and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when KAMA turns down OR loses weekly EMA34 alignment
            if kama_direction == -1 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when KAMA turns up OR loses weekly EMA34 alignment
            if kama_direction == 1 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_WeeklyEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0