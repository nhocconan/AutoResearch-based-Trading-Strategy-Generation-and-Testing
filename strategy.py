#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Volume_Spike_Trend_v1
Hypothesis: On 12h timeframe, price touching Camarilla R3/S3 levels from prior 1d combined with volume spike (>2.0x 20-period average) and 1d EMA34 trend filter captures high-probability reversals in ranging markets and continuations in trending markets. 
In bull 1d regime (close > EMA34), favor longs at S3; in bear 1d regime (close < EMA34), favor shorts at R3. 
Volume confirmation reduces false signals. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels and trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d Camarilla levels (based on prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    camarilla_r3 = typical_price + (1.1 * range_1d / 2)
    camarilla_s3 = typical_price - (1.1 * range_1d / 2)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 12h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 4 days (8 * 12h = 96h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i]) if 'atr' in locals() else True or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Calculate 12h ATR for stoploss
        if i >= 100:  # ensure we have enough data for ATR
            high = prices['high'].values
            low = prices['low'].values
            close = prices['close'].values
            
            tr1 = pd.Series(high - low)
            tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
            tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=10, min_periods=10).mean().values
        else:
            atr = np.full(n, np.nan)
        
        if np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # 1d trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price touches or breaks below S3 (mean reversion)
                long_condition = (price <= s3_level) and vol_conf
            else:  # bear regime
                # Bear regime: short when price touches or breaks above R3 (mean reversion)
                short_condition = (price >= r3_level) and vol_conf
            
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

name = "12h_Camarilla_Pivot_Volume_Spike_Trend_v1"
timeframe = "12h"
leverage = 1.0