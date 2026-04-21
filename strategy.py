#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dATR_Trend_v1
Hypothesis: On 12h timeframe, price breaking above/below Camarilla R3/S3 levels from 1d timeframe, 
with 1d ATR-based trend filter (price > EMA34) and volume confirmation (volume > 1.5x 20-period average) 
captures strong directional moves with reduced whipsaw. 
In bull trend (1d close > 1d EMA34), favor longs when price breaks R3; 
in bear trend (1d close < 1d EMA34), favor shorts when price breaks S3. 
Volume confirmation ensures institutional participation. Discrete sizing (0.25) minimizes fee churn. 
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla and EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d Camarilla pivot levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_pivot = typical_price
    camarilla_r3 = camarilla_pivot + (range_1d * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (range_1d * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 12h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 4 days (8 * 12h = 96h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = prices['close'].iloc[i]
        daily_ema = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull trend: long when price breaks above R3
                long_condition = (price > r3_level) and vol_conf
            else:  # bear trend
                # Bear trend: short when price breaks below S3
                short_condition = (price < s3_level) and vol_conf
            
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
            
            # Check stoploss (2.5x ATR based on 12h ATR)
            high_12h = prices['high'].iloc[i]
            low_12h = prices['low'].iloc[i]
            close_12h = prices['close'].iloc[i]
            
            # Calculate 12h ATR for stoploss (using 14-period)
            if i >= 14:
                tr1 = high_12h - low_12h
                tr2 = np.abs(high_12h - prices['close'].iloc[i-1])
                tr3 = np.abs(low_12h - prices['close'].iloc[i-1])
                atr_12h = max(tr1, tr2, tr3)
                # Simple ATR approximation for stop (more accurate would use rolling)
                if i >= 100:  # After warmup, use recent ATR
                    atr_values = []
                    for j in range(max(0, i-13), i+1):
                        tr1_j = prices['high'].iloc[j] - prices['low'].iloc[j]
                        tr2_j = np.abs(prices['high'].iloc[j] - prices['close'].iloc[j-1]) if j > 0 else 0
                        tr3_j = np.abs(prices['low'].iloc[j] - prices['close'].iloc[j-1]) if j > 0 else 0
                        atr_values.append(max(tr1_j, tr2_j, tr3_j))
                    atr_12h = np.mean(atr_values[-14:]) if len(atr_values) >= 14 else np.mean(atr_values)
                else:
                    atr_12h = 0
            else:
                atr_12h = 0
            
            if position == 1:
                if price < entry_price - 2.5 * atr_12h:
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
                if price > entry_price + 2.5 * atr_12h:
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

name = "12h_Camarilla_R3_S3_Breakout_1dATR_Trend_v1"
timeframe = "12h"
leverage = 1.0