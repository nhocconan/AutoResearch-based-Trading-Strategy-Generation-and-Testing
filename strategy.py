#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrendRegime_VolumeSpike_v2
Hypothesis: On 12h timeframe, Camarilla R3/S3 levels from daily pivot act as strong support/resistance. 
Breakout above R3 with volume > 2.0x 20-period average and daily close > daily EMA34 (bull regime) triggers long.
Breakdown below S3 with volume confirmation and daily close < daily EMA34 (bear regime) triggers short.
ATR-based stoploss (2.5x ATR) and max hold of 8 bars (4 days) controls risk. Discrete sizing (0.25) minimizes fee churn.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily trend and Camarilla levels)
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
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    camarilla_range = high_1d - low_1d
    r3 = pivot + (camarilla_range * 1.1 / 4.0)  # R3 = pivot + 1.1*(H-L)/4
    s3 = pivot - (camarilla_range * 1.1 / 4.0)  # S3 = pivot - 1.1*(H-L)/4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (volume > 2.0x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 4 days (8 * 12h = 96h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        daily_ema = ema_34_1d_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Daily trend regime
        is_bull = close_1d[-1] > daily_ema if len(close_1d) > 0 else False  # Simplified: use current close vs EMA
        is_bear = close_1d[-1] < daily_ema if len(close_1d) > 0 else False
        # Better approach: use aligned daily EMA vs price
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R3 with volume
                long_condition = (price > r3_level) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S3 with volume
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
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
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
                if price > entry_price + 2.5 * atr[i]:
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

name = "12h_Camarilla_R3_S3_Breakout_1dTrendRegime_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0