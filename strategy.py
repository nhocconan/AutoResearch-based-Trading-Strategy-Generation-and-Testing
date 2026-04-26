#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike (>2.0x average) on 12h timeframe captures institutional breakouts with low whipsaw. Discrete sizing (0.25) and ATR(14) stoploss (2.0x) target ~12-37 trades/year. Works in bull/bear by only taking breakouts aligned with 1d trend.
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Camarilla levels from previous 1d bar
    # Calculate from previous completed 1d bar (avoid look-ahead)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Camarilla R1, S1, R3, S3
    camarilla_r1 = ((prev_high_1d - prev_low_1d) * 1.1 / 12) + prev_close_1d
    camarilla_s1 = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 12)
    camarilla_r3 = ((prev_high_1d - prev_low_1d) * 1.1 / 4) + prev_close_1d
    camarilla_s3 = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of EMA34 (34), ATR (14), volume MA (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema34_1d_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        r1_val = r1_12h[i]
        s1_val = s1_12h[i]
        r3_val = r3_12h[i]
        s3_val = s3_12h[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(r1_val) or np.isnan(s1_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA34 = uptrend, price < 1d EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Camarilla R1/S1 breakout in direction of 1d trend + volume
        long_entry = (close_val > r1_val) and is_uptrend and vol_conf
        short_entry = (close_val < s1_val) and is_downtrend and vol_conf
        
        # Exit conditions: ATR-based stoploss or Camarilla S3/R3 touch
        long_exit = False
        short_exit = False
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or close_val < s3_val  # Stop or Camarilla S3
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or close_val > r3_val  # Stop or Camarilla R3
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Approximate entry price for stop calculation
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0