#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: On 4h timeframe, use Camarilla R1/S1 levels from 1d for breakout entries, filtered by 1d trend direction (close > EMA34) and volume spike (>2.0x 20-period average). Exit via ATR-based trailing stop (3*ATR from extreme) or opposite Camarilla level break. Designed for 20-50 trades/year on 4h by requiring daily alignment and volume confirmation, reducing overtrading while capturing structured moves in both bull and bear markets. ATR stoploss controls drawdown during volatile periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_1d_high - prev_1d_low
    r1 = prev_1d_close + 1.1 * camarilla_range / 6
    s1 = prev_1d_close - 1.1 * camarilla_range / 6
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ATR for trailing stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0
    short_low = 0.0
    
    # Warmup: need 1d EMA warmup, volume MA warmup, ATR warmup
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike
            long_signal = (close[i] > r1_aligned[i]) and trend_1d_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 1d downtrend + volume spike
            short_signal = (close[i] < s1_aligned[i]) and trend_1d_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_high = high[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_low = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update highest high for trailing stop
            long_high = max(long_high, high[i])
            # Exit: ATR trailing stop (3*ATR from high) OR price breaks below S1 OR 1d trend turns down
            if (close[i] < long_high - 3.0 * atr[i] or 
                close[i] < s1_aligned[i] or 
                not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update lowest low for trailing stop
            short_low = min(short_low, low[i])
            # Exit: ATR trailing stop (3*ATR from low) OR price breaks above R1 OR 1d trend turns up
            if (close[i] > short_low + 3.0 * atr[i] or 
                close[i] > r1_aligned[i] or 
                not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0