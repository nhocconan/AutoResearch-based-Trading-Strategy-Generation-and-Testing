#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v1
Hypothesis: Use 12h timeframe for lower trade frequency (target: 50-150 trades over 4 years). 
Breakout of 1d Camarilla R1/S1 levels with 1d EMA34 trend filter and volume confirmation (>2.0x 20-period average). 
ATR-based stoploss (2.5x) and trend reversal exit. Designed to work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla, EMA34, volume, ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d indicators ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend regime
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR (14-period) for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d volume confirmation (volume > 2.0x 20-period average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirmed_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d.astype(float))
    
    # === 1d Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = prev_low_1d[0] = prev_close_1d[0] = np.nan  # first bar invalid
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1_1d = pivot_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    s1_1d = pivot_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 12h close price for breakout detection ===
    close_12h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_confirmed_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close_12h[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        vol_conf = bool(volume_confirmed_1d_aligned[i])  # Convert to bool
        
        # Trend alignment: price above EMA34 for long, below for short
        uptrend = price > ema_34_1d_val
        downtrend = price < ema_34_1d_val
        
        if position == 0:
            # Long: price closes above R1, uptrend, volume confirmed
            long_condition = (price > r1_val) and uptrend and vol_conf
            # Short: price closes below S1, downtrend, volume confirmed
            short_condition = (price < s1_val) and downtrend and vol_conf
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below EMA34)
                elif price < ema_34_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above EMA34)
                elif price > ema_34_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0