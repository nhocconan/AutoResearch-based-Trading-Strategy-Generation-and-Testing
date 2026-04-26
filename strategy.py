#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4h1dTrend_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts from 1h data with 4h and 1d EMA trend filters (price > EMA on both = uptrend, price < EMA on both = downtrend) and volume confirmation (2.0x average). Uses ATR trailing stop (2.0) for risk management. Designed for low trade frequency (~15-35/year) by requiring strong confluence: breakout + multi-HTF trend + volume spike. Works in bull markets (breakouts with trend) and bear markets (short breakdowns against trend). Session filter (08-20 UTC) reduces noise trades.
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
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h and 1d data for HTF filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA(34) for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Previous 1h bar's high, low, close for Camarilla levels (using 1h data resampled internally)
    # Since we're on 1h timeframe, use previous bar for Camarilla calculation
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # Calculate Camarilla levels: R1, S1 from previous 1h bar
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align 4h and 1d indicators to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period)
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
    
    # Warmup: max of 4h EMA (34), 1d EMA (34), volume MA (20), ATR (14)
    start_idx = max(34, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1[i]) or 
            np.isnan(S1[i]) or 
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
        
        ema_34_4h_val = ema_34_4h_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        R1_val = R1[i]
        S1_val = S1[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above R1, uptrend (close > EMA on both 4h and 1d), volume spike
            long_signal = (high_val > R1_val) and (close_val > ema_34_4h_val) and (close_val > ema_34_1d_val) and (volume_val > 2.0 * vol_ma_val)
            # Short: break below S1, downtrend (close < EMA on both 4h and 1d), volume spike
            short_signal = (low_val < S1_val) and (close_val < ema_34_4h_val) and (close_val < ema_34_1d_val) and (volume_val > 2.0 * vol_ma_val)
            
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
            # Exit: trailing stop hit or trend reversal (close < EMA on either 4h or 1d)
            if (low_val < long_stop) or (close_val < ema_34_4h_val) or (close_val < ema_34_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA on either 4h or 1d)
            if (high_val > short_stop) or (close_val > ema_34_4h_val) or (close_val > ema_34_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4h1dTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0