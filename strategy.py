#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakout in the direction of 12h EMA50 trend with volume confirmation (>1.3x 20-period MA) captures high-probability trend continuation moves. Camarilla levels derived from daily OHLC act as institutional support/resistance. R3/S3 represents the first significant breakout level where institutional interest increases. Volume confirmation ensures breakout validity. 12h EMA50 provides trend filter to avoid counter-trend trades. Discrete position sizing (±0.25) and ATR-based trailing stop (2.0x) for exits. Targets 15-25 trades/year by requiring trend alignment, volume confirmation, and Camarilla breakout structure—designed to work in both bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend) markets with BTC/ETH edge from Camarilla's mathematical derivation of institutional order flow zones.
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
    
    # Load daily data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3, R4, S4
    # Camarilla formula: Close + (High-Low) * multiplier
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + daily_range * 1.1000  # R3 level
    camarilla_s3 = close_1d - daily_range * 1.1000  # S3 level
    camarilla_r4 = close_1d + daily_range * 1.2666  # R4 level (strong breakout)
    camarilla_s4 = close_1d - daily_range * 1.2666  # S4 level (strong breakdown)
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_6h_values = atr_6h.values
    
    # Volume confirmation: volume > 1.3 * 20-period MA on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20), daily Camarilla needs 1 day
    start_idx = max(50, 14, 20) + 2  # +2 to ensure 1 day of 6h data for daily levels
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr_6h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(r4_val) or np.isnan(s4_val) or 
            np.isnan(ema_trend) or np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        trend_bullish = close_val > ema_trend
        trend_bearish = close_val < ema_trend
        
        # Camarilla breakout conditions: price breaks R3/S3 with trend alignment + volume confirmation
        # Using R3/S3 for entry, R4/S4 as confirmation of strong breakout
        long_breakout = close_val > r3_val
        short_breakout = close_val < s3_val
        
        # Additional confirmation: strong breakout beyond R4/S4 increases validity
        long_confirmation = close_val > r4_val  # Price beyond R4 = strong bullish breakout
        short_confirmation = close_val < s4_val  # Price below S4 = strong bearish breakdown
        
        long_entry = trend_bullish and long_breakout and vol_conf
        short_entry = trend_bearish and short_breakout and vol_conf
        
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
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
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

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0