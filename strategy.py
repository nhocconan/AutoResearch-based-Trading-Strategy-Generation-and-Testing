#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation (volume > 1.8x 20-period average) captures institutional moves in both bull and bear markets. 
In bull trend (price > 1d EMA34), go long on R1 breakout with volume. In bear trend (price < 1d EMA34), go short on S1 breakdown with volume. 
Discrete sizing (0.25) minimizes fee churn. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Camarilla pivot levels (R1, S1) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot and levels
    pivot = (high + low + close) / 3
    range_hl = high - low
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # === 12h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 4 days (8 * 12h = 96h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pivot[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            if is_bull:
                # Bull trend: long on R1 breakout with volume
                long_condition = (price > R1[i]) and vol_conf
            else:  # bear trend
                # Bear trend: short on S1 breakdown with volume
                short_condition = (price < S1[i]) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR approximation using range)
            atr_approx = pd.Series(range_hl).rolling(window=14, min_periods=14).mean().values[i]
            if position == 1:
                if price < entry_price - 2.5 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrendRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0