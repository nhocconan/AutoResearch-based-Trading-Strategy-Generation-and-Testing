#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike confirmation (>2.0x 20-period MA). Only takes trades in direction of 4h trend to avoid counter-trend whipsaws. Uses discrete position sizing (0.20) and session filter (08-20 UTC) to minimize fee drag. Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
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
    
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h EMA50 for HTF trend regime ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d volume confirmation (2.0x 20-period MA) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1h price data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 1h Camarilla pivot levels (R1, S1) ===
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
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        vol_ma_20_1d_val = vol_ma_20_1d_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        
        # Volume confirmation: current 1h volume > 2.0x 1d average volume (scaled)
        # Approximate 1d volume by 1h volume * 24 (since 24x 1h bars in 1d)
        volume_confirm = volume_now > 2.0 * (vol_ma_20_1d_val / 24.0)
        
        # Trend alignment: price above/below 4h EMA50
        uptrend = price > ema_50_4h_val
        downtrend = price < ema_50_4h_val
        
        if position == 0:
            # Long: price breaks above R1, uptrend alignment, volume confirm
            long_condition = (price > r1_val) and uptrend and volume_confirm
            # Short: price breaks below S1, downtrend alignment, volume confirm
            short_condition = (price < s1_val) and downtrend and volume_confirm
            
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
            
            # Minimum holding period of 6 bars to reduce churn (~6 hours)
            if bars_since_entry < 6:
                signals[i] = 0.20 if position == 1 else -0.20
                continue
            
            # Exit conditions: trend reversal or time-based exit (max 24 bars = 1 day)
            if position == 1:
                if price < ema_50_4h_val or bars_since_entry >= 24:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price > ema_50_4h_val or bars_since_entry >= 24:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0