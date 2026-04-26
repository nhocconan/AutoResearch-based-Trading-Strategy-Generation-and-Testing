#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, price breaking above Camarilla R1 level with 4h EMA20 uptrend and volume spike captures high-probability long entries; breaking below S1 with 4h EMA20 downtrend and volume spike captures short entries. Uses 4h for signal direction, 1h only for entry timing. Adds UTC 08-20 session filter to avoid low-liquidity hours. Uses discrete position sizing (±0.20) to minimize fee churn. Targets 15-37 trades/year by requiring HTF trend alignment, volume confirmation, and precise pivot structure. Designed to work in both bull (trend continuation) and bear (trend continuation down) markets with BTC/ETH edge from Camarilla pivot effectiveness.
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
    
    # Pre-compute session hours (08-20 UTC) once before loop
    # open_time is already datetime64[ms], so .index.hour works directly
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend
    close_4h = pd.Series(df_4h['close'])
    ema_20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Daily data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (look-ahead safe)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but will be filtered by min_periods later
    
    rangeprev = high_1d_prev - low_1d_prev
    R1 = close_1d_prev + rangeprev * 1.1 / 12
    S1 = close_1d_prev - rangeprev * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h = tr_1h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_1h_values = atr_1h.values
    
    # Volume spike filter: volume > 1.8 * 20-period MA on 1h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA20 (20), ATR (14), volume MA (20), Camarilla needs 2 days
    start_idx = max(20, 14, 20) + 24  # +24 to ensure 2 days of 1h data for Camarilla
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
            lowest_since_short = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        ema_trend = ema_20_4h_aligned[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema_trend) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA20, bearish when price < EMA20
        trend_bullish = close_val > ema_trend
        trend_bearish = close_val < ema_trend
        
        # Breakout conditions: price breaks R1/S1 with trend alignment + volume spike
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0