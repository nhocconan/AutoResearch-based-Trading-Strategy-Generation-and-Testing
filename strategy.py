#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_1dVolumeSpike_ATRStop
Hypothesis: Trade Camarilla R1/S1 breakouts from 1d data with 1w EMA50 trend filter (price > EMA50 = uptrend) and 1d volume confirmation (2.5x average). Uses ATR trailing stop (2.0) for risk management. Designed for low trade frequency (~15-30/year) by requiring strong confluence: breakout + HTF trend + volume spike. Works in bull markets (breakouts with trend) and bear markets (short breakdowns against trend). Focus on BTC/ETH as primary targets.
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
    
    # Get 1d and 1w data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Previous 1d bar's high, low, close for Camarilla levels
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1 from 1d data
    camarilla_range_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + camarilla_range_1d * 1.0/12
    S1_1d = prev_close_1d - camarilla_range_1d * 1.0/12
    
    # 1d volume MA for confirmation
    vol_ma_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR for stop (14-period on 4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1w EMA (50), 1d volume MA (20), 4h ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        R1_val = R1_1d_aligned[i]
        S1_val = S1_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above R1, uptrend (close > EMA50_1w), volume spike
            long_signal = (high_val > R1_val) and (close_val > ema_50_1w_val) and (volume_val > 2.5 * vol_ma_1d_val)
            # Short: break below S1, downtrend (close < EMA50_1w), volume spike
            short_signal = (low_val < S1_val) and (close_val < ema_50_1w_val) and (volume_val > 2.5 * vol_ma_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50_1w)
            if (low_val < long_stop) or (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50_1w)
            if (high_val > short_stop) or (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_1dVolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0