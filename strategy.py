#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSqueeze
Hypothesis: Daily Camarilla H3/L3 breakouts with weekly EMA50 trend filter and volume squeeze confirmation capture institutional order flow on higher timeframes. Works in bull markets (breakouts with trend) and bear markets (mean reversion at H3/L3 levels). Target trade frequency: 10-25/year to minimize fee drag. Uses 1d primary timeframe with 1w HTF for trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1w EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d ATR for volatility filter
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels: H3, L3
    camarilla_range = 1.1 * (prev_high - prev_low)
    h3 = prev_close + camarilla_range * 0.275
    l3 = prev_close - camarilla_range * 0.275
    
    # Align Camarilla levels to 1d timeframe (already completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume squeeze: current volume < 0.5 * 20-period average (low volatility breakout setup)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_squeeze = volume < (vol_ma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA (50) + volume MA (20) + ATR (14) + Camarilla (2)
    start_idx = max(50, 20, 14, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume squeeze + 1w EMA50 trend alignment
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            
            long_entry = long_breakout and volume_squeeze[i] and (curr_close > ema_50_1w_aligned[i])
            short_entry = short_breakout and volume_squeeze[i] and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below H3 (failed breakout) or trend turns bearish
            if curr_close < h3_aligned[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L3 (failed breakout) or trend turns bullish
            if curr_close > l3_aligned[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSqueeze"
timeframe = "1d"
leverage = 1.0