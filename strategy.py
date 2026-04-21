#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation (>2.0x 20-period MA).
Long when price breaks above R1, above 4h EMA50, and volume > 2.0x average.
Short when price breaks below S1, below 4h EMA50, and volume > 2.0x average.
Uses 4h ATR (14-period) for stoploss (2.0x) and minimum holding period of 3 bars.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Uses 4h for signal direction (trend regime), 1h only for entry timing precision.
Session filter: 08-20 UTC to reduce noise trades.
Position size: 0.20 (discrete level to minimize fee churn).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend regime)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend regime ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 4h ATR (14-period) for stoploss ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h_arr, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h_arr, 1)))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 1h price and volume ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Volume confirmation (2.0x 20-period MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1h Camarilla pivot levels (R1, S1) ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ms], access .hour via index
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        vol_avg = vol_ma[i]
        r1_val = r1[i]
        s1_val = s1[i]
        
        # Volume confirmation: current volume > 2.0x average (tight threshold)
        volume_confirm = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1, above 4h EMA50, volume confirm, session OK
            long_condition = (price > r1_val) and (price > ema_50_4h_val) and volume_confirm
            # Short: price breaks below S1, below 4h EMA50, volume confirm, session OK
            short_condition = (price < s1_val) and (price < ema_50_4h_val) and volume_confirm
            
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
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.20 if position == 1 else -0.20
                continue
            
            # Check stoploss (2.0x ATR from 4h, aligned)
            if position == 1:
                if price < entry_price - 2.0 * atr_4h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 4h EMA50)
                elif price < ema_50_4h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price > entry_price + 2.0 * atr_4h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 4h EMA50)
                elif price > ema_50_4h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0