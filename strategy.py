#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4h1dTrend_VolumeSpike_v2
Hypothesis: Trade Camarilla R1/S1 level breakouts on 1h with 4h EMA20 and 1d EMA50 trend filters plus volume confirmation (1.8x average). Uses ATR trailing stop (2.0). Designed for moderate trade frequency (~15-30/year) by requiring confluence: breakout at pivot level + multi-HTF trend + volume spike. Works in bull markets (breakouts with trend) and bear markets (short breakdowns against trend). Session filter (08-20 UTC) reduces noise. Position size 0.20.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for HTF trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R1/S1)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) * 1.0 / 4
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) * 1.0 / 4
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 1.8x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on 1h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 4h EMA (20), 1d EMA (50), volume MA (20), 1h ATR (14)
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_20_4h_val = ema_20_4h_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above R1, uptrend (close > both EMAs), volume spike
            long_signal = (high_val > camarilla_r1_val) and (close_val > ema_20_4h_val) and (close_val > ema_50_1d_val) and (volume_val > 1.8 * vol_ma_val)
            # Short: break below S1, downtrend (close < both EMAs), volume spike
            short_signal = (low_val < camarilla_s1_val) and (close_val < ema_20_4h_val) and (close_val < ema_50_1d_val) and (volume_val > 1.8 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < 4h EMA)
            if (low_val < long_stop) or (close_val < ema_20_4h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > 4h EMA)
            if (high_val > short_stop) or (close_val > ema_20_4h_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4h1dTrend_VolumeSpike_v2"
timeframe = "1h"
leverage = 1.0