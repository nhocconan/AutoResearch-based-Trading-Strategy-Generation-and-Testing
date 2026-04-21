#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation (>2.0x 20-period average). Uses discrete sizing (0.25) and ATR-based stop (2.0x). Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag. Works in bull/bear via 1d trend filter: long only in uptrend (price > EMA50), short only in downtrend (price < EMA50). Avoids overtrading by requiring volume spike and strict trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for HTF trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h close ===
    close = prices['close'].values
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 6h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        # Trend alignment: price above EMA50 for long, below for short
        uptrend = price > ema_50_1d_val
        downtrend = price < ema_50_1d_val
        
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
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0