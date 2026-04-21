#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume spike capture institutional moves with confirmation. Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise. Target 60-150 trades over 4 years (15-37/year) to minimize fee drag. Works in bull/bear by requiring 4h trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1h Camarilla levels from prior 1h session (HLC of previous 1h bar) ===
    # We need to compute 1h Camarilla from 1h data, but prices is 1h OHLCV
    # So we can use prices directly for 1h Camarilla calculation
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # Shift by 1 to get previous bar's HLC for current bar's levels
    prev_high_1h = np.roll(high_1h, 1)
    prev_low_1h = np.roll(low_1h, 1)
    prev_close_1h = np.roll(close_1h, 1)
    prev_high_1h[0] = high_1h[0]  # first bar uses its own high
    prev_low_1h[0] = low_1h[0]
    prev_close_1h[0] = close_1h[0]
    
    camarilla_r1_1h = prev_close_1h + (prev_high_1h - prev_low_1h) * 1.1 / 12
    camarilla_s1_1h = prev_close_1h - (prev_high_1h - prev_low_1h) * 1.1 / 12
    
    # === 4h trend filter: 50-period EMA on 4h ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === Volume spike filter (20-period on 1h) ===
    volume_1h = prices['volume'].values
    vol_ma_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1h = volume_1h / vol_ma_1h
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ms], use DatetimeIndex hour
    hours = prices.index.hour  # pre-computed DatetimeIndex hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_1h[i]) or
            np.isnan(camarilla_r1_1h[i]) or np.isnan(camarilla_s1_1h[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_1h[i]
        r1 = camarilla_r1_1h[i]
        s1 = camarilla_s1_1h[i]
        vol_spike = vol_ratio_1h[i]
        trend_4h = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 4h EMA50 (bullish trend)
            if price_close > r1 and vol_spike > 2.0 and price_close > trend_4h:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume spike > 2.0 + price below 4h EMA50 (bearish trend)
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend_4h:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: reverse signal or loss of trend/volume
            if position == 1:
                # Exit long if price breaks below S1 or trend turns bearish
                if price_close < s1 or price_close < trend_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short if price breaks above R1 or trend turns bullish
                if price_close > r1 or price_close > trend_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0