#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakout from 1d combined with 1d EMA34 trend filter and volume confirmation (>2.0x 20-period average) captures institutional breakouts with reduced whipsaw. Works in bull/bear regimes by only taking breakouts in direction of 1d trend. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA34 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d_arr + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d_arr - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 6  # max 1 day (6 * 4h = 24h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = prices['close'].iloc[i]
        weekly_ema = ema_34_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R1
                long_condition = (price > r1) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S1
                short_condition = (price < s1) and vol_conf
            
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
            
            # Check stoploss (2.0x ATR) - using 4h ATR
            high_4h = prices['high'].iloc[i]
            low_4h = prices['low'].iloc[i]
            close_4h = prices['close'].iloc[i]
            if i > 0:
                prev_close = prices['close'].iloc[i-1]
                tr = max(high_4h - low_4h, abs(high_4h - prev_close), abs(low_4h - prev_close))
            else:
                tr = high_4h - low_4h
            
            # Calculate ATR using rolling window
            if i >= 14:
                atr_vals = []
                for j in range(max(0, i-14), i+1):
                    h = prices['high'].iloc[j]
                    l = prices['low'].iloc[j]
                    c_prev = prices['close'].iloc[j-1] if j > 0 else prices['close'].iloc[j]
                    tr_j = max(h - l, abs(h - c_prev), abs(l - c_prev))
                    atr_vals.append(tr_j)
                atr = np.mean(atr_vals[-14:]) if len(atr_vals) >= 14 else np.mean(atr_vals) if atr_vals else 0
            else:
                atr = 0
            
            if position == 1:
                if price < entry_price - 2.0 * atr:
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
                if price > entry_price + 2.0 * atr:
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

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0