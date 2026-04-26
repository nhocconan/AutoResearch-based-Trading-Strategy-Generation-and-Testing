#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On 4h timeframe, price breaking Camarilla R3/S3 levels in the direction of 1w EMA50 trend with volume confirmation (>1.8x 20-period MA) captures institutional trend continuation moves. The 1w EMA50 acts as a strong trend filter, while Camarilla R3/S3 levels represent significant intraday resistance/support. Volume spike confirms participation. Designed for 20-50 trades/year with discrete sizing (±0.25) and ATR-based trailing stop (2.5x) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
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
    
    # Load 1w data ONCE before loop for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_series = pd.Series(df_1w['close'].values)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + daily_range * 1.1 / 4
    camarilla_s3 = close_1d - daily_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 4h ATR(20) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_4h_values = atr_4h.values
    
    # Volume spike filter: volume > 1.8 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA (50), ATR (20), volume MA (20) + time for 1d alignment
    start_idx = max(50, 20, 20) + 4  # +4 to ensure 1d bar completion (4h -> 1d: 6 bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Camarilla breakout conditions: price breaks R3/S3 with trend alignment + volume spike
        long_breakout = close_val > r3_val
        short_breakout = close_val < s3_val
        
        long_entry = trend_bullish and long_breakout and vol_spike
        short_entry = trend_bearish and short_breakout and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0