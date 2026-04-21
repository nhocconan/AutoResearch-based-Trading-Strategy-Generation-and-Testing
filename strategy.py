#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeConfirm_v1
Hypothesis: On 1h timeframe, Camarilla pivot R1/S1 breakouts with 4h EMA50 trend filter and 1d volume confirmation capture institutional breakouts with controlled frequency. 
Use 4h for trend direction (EMA50) and 1d for volume regime (volume > 1.5x 20-day average). 
Only take longs in 4h uptrend when price breaks above R1 with volume confirmation, and shorts in 4h downtrend when price breaks below S1 with volume confirmation. 
Session filter (08-20 UTC) reduces noise. Discrete sizing (0.20) minimizes fee churn. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, 1d for volume)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend direction ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d volume confirmation (volume > 1.5x 20-day average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_conf_1d = volume_1d > (1.5 * vol_ma_20_1d)
    vol_conf_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_conf_1d.astype(float))
    
    # === Calculate Camarilla pivots from previous day (using 1d OHLC) ===
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Align Camarilla levels to 1h timeframe (previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ns] in prices DataFrame
    hours = prices.index.hour  # Pre-compute before loop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 6  # max 6 hours (6 * 1h = 6h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_conf_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = vol_conf_1d_aligned[i] > 0.5  # Convert back to boolean
        
        # 4h trend regime
        is_uptrend = price > ema_50_4h_val
        is_downtrend = price < ema_50_4h_val
        
        if position == 0:
            if in_session:
                # Uptrend: long when price breaks above R1 with volume confirmation
                long_condition = (price > r1_val) and vol_conf
                # Downtrend: short when price breaks below S1 with volume confirmation
                short_condition = (price < s1_val) and vol_conf
                
                if is_uptrend and long_condition:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif is_downtrend and short_condition:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0