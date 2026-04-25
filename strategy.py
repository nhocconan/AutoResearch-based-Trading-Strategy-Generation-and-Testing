#!/usr/bin/env python3
"""
1h Camarilla R1S1 Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: On 1h timeframe, Camarilla R1/S1 levels provide precise intraday entry points when aligned with 4h trend (EMA50) and confirmed by volume spikes. This captures momentum continuations in both bull and bear markets while minimizing false breakouts. Using 1h timeframe targets 60-150 total trades over 4 years (15-37/year) to control fee drift, with 4h EMA50 filtering for higher timeframe trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (using daily pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate R1 and S1 for each 1d bar
    rng = high_1d - low_1d
    r1 = close_1d + 1.1 * (rng / 12)  # Camarilla R1 = Close + 1.1*(Range/12)
    s1 = close_1d - 1.1 * (rng / 12)  # Camarilla S1 = Close - 1.1*(Range/12)
    
    # Align to 1h timeframe (use previous day's levels, so shift by 1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=1)
    
    # Calculate ATR for stoploss (using 1h data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50 and ATR warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Session filter: 08-20 UTC (reduce noise outside active trading hours)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above R1 AND above 4h EMA50 (uptrend filter) AND volume spike AND in session
            long_condition = (curr_close > r1_level) and (curr_close > ema_trend) and volume_spike and in_session
            # Short: price breaks below S1 AND below 4h EMA50 (downtrend filter) AND volume spike AND in session
            short_condition = (curr_close < s1_level) and (curr_close < ema_trend) and volume_spike and in_session
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Check stoploss: 2.0 * ATR below entry
            if curr_close <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit long: price returns below R1 or trend breaks
            elif curr_close <= r1_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Check stoploss: 2.0 * ATR above entry
            if curr_close >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit short: price returns above S1 or trend breaks
            elif curr_close >= s1_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0