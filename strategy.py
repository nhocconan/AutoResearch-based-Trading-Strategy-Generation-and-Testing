#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: 4h breakouts at 1d Camarilla R3/S3 levels with 1d EMA34 trend filter, volume confirmation (>2x 20-bar MA), and ATR-based trailing stop. Uses discrete position sizing (0.25) to minimize fee churn. Designed for lower frequency (target 20-50 trades/year) to avoid fee drag, works in bull/bear via trend alignment.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1d ATR for volatility and stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = 14
    atr_1d = np.zeros(len(df_1d))
    if len(df_1d) >= atr_period:
        atr_1d[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of calculations
    start_idx = max(20, 34, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions
        long_entry = (close_val > r3_val) and bullish_1d and vol_spike
        short_entry = (close_val < s3_val) and bearish_1d and vol_spike
        
        # Update tracking variables
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
        elif position == 0:
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        # ATR-based trailing stop (2.5 * ATR)
        stop_long = highest_since_entry - 2.5 * atr_val if position == 1 else np.inf
        stop_short = lowest_since_entry + 2.5 * atr_val if position == -1 else -np.inf
        
        # Stoploss conditions
        stop_long_hit = position == 1 and low[i] < stop_long
        stop_short_hit = position == -1 and high[i] > stop_short
        
        # Exit conditions: stoploss hit or trend reversal
        if long_entry and position != 1 and not stop_long_hit:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_entry = high[i]
        elif short_entry and position != -1 and not stop_short_hit:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            lowest_since_entry = low[i]
        elif position == 1 and (stop_long_hit or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (stop_short_hit or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0