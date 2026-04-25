#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume spike confirmation, and chop regime filter.
Targets 12-30 trades/year by requiring: 1) price breaks R3/S3 levels (stronger breakout than R1/S1), 2) aligned with 1d EMA50 trend, 3) volume > 2.5x 20-period average, 4) choppy market filter (Chop > 61.8) to avoid whipsaw in ranging markets.
Designed for low turnover and high edge in both bull/bear markets via trend alignment, institutional volume confirmation, and regime filtering.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA50 trend filter and Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R3 and S3 levels (stronger breakout levels)
    R3 = prev_close + 1.1 * prev_range * (3.0/8.0)  # R3 = C + 1.1*(3HL/8)
    S3 = prev_close - 1.1 * prev_range * (3.0/8.0)  # S3 = C - 1.1*(3HL/8)
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 2.5 * 20-period average (strict spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.5)
    
    # Chop regime filter: avoid trading in strong trends (Chop < 38.2) or choppy markets (Chop > 61.8) - actually we want to avoid strong trends
    # Chop > 61.8 = ranging market (good for mean reversion, but we're doing breakout)
    # Chop < 38.2 = trending market (good for breakout)
    # We want to trade breakouts in trending markets, so Chop < 38.2
    high_low = pd.Series(high - low).rolling(window=14, min_periods=14).values
    true_range = np.maximum(
        high_low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    true_range[0] = high[0] - low[0]  # first value
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(high_low.sum() / (atr_14 * 14)) / np.log10(10)
    chop_ma = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    chop_filter = chop_ma < 38.2  # trending market good for breakout
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA50 (50) and indicators (14)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment, and chop filter
            # Long breakout: price breaks above R3 with uptrend, volume confirmation, and trending market
            long_breakout = (curr_close > R3_aligned[i]) and uptrend and volume_confirm[i] and chop_filter[i]
            # Short breakout: price breaks below S3 with downtrend, volume confirmation, and trending market
            short_breakout = (curr_close < S3_aligned[i]) and downtrend and volume_confirm[i] and chop_filter[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S3 (mean reversion) or trend changes or chop increases (range developing)
            elif curr_close < S3_aligned[i] or not uptrend or chop_ma[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Calculate 6h ATR (same as above)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R3 (mean reversion) or trend changes or chop increases (range developing)
            elif curr_close > R3_aligned[i] or not downtrend or chop_ma[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0