#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike_v1
Hypothesis: On 1h timeframe, use 4h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume confirmation (>2.0x 20-bar MA). 
In bull regime (price > 4h EMA50), take longs on R1 breakouts; in bear regime (price < 4h EMA50), take shorts on S1 breakdowns. 
Volume confirmation from 1d timeframe ensures institutional participation and reduces false breakouts. 
Discrete sizing (0.20) minimizes fee churn. ATR-based stoploss (2.0x) and pivot-based exits (S1 for longs, R1 for shorts) control risk.
Target: 80-120 total trades over 4 years by requiring confluence of 4h breakout, 4h trend, and 1d volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for Camarilla/EMA, 1d for volume)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend regime ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d volume confirmation (volume > 2.0x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_conf_1d = vol_1d > (2.0 * vol_ma_20_1d)
    vol_conf_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_conf_1d.astype(float))
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 4h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = prev_low_4h[0] = prev_close_4h[0] = np.nan
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    r1_4h = pivot_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12.0
    s1_4h = pivot_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12.0
    
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(vol_conf_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Session filter
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
        r1_val = r1_4h_aligned[i]
        s1_val = s1_4h_aligned[i]
        vol_conf = vol_conf_1d_aligned[i] > 0.5  # boolean
        
        # Trend regime
        is_bull = price > ema_50_4h_val
        is_bear = price < ema_50_4h_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > r1_val) and vol_conf
                short_condition = (price < s1_val) and vol_conf and (price < ema_50_4h_val * 0.995)
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < s1_val) and vol_conf
                long_condition = (price > r1_val) and vol_conf and (price > ema_50_4h_val * 1.005)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
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
                # Exit if price breaks below S1 (failed breakout)
                elif price < s1_val:
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
                # Exit if price breaks above R1 (failed breakdown)
                elif price > r1_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike_v1"
timeframe = "1h"
leverage = 1.0