#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume regime filter.
Targets 20-40 trades/year by requiring: 1) price breaks daily R1/S1 levels, 2) aligned with 1d EMA50 trend,
3) volume > 1.8x 30-period average. Uses chop filter (BW < 50th percentile) to avoid ranging markets.
Designed to work in both bull and bear markets by following the 1d trend direction, avoiding counter-trend
entries that fail in ranging/volatile conditions. Volume regime reduces false breakouts while chop filter
ensures trades occur in trending environments only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots and EMA50 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 1d levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA50 trend filter (loaded ONCE)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume regime: current volume > 1.8 * 30-period average (regime filter)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_regime = volume > (vol_ma * 1.8)
    
    # Bollinger Width for chop regime (20,2) - avoid ranging markets
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2 * dev)
    lower_band = basis - (2 * dev)
    bb_width = (upper_band - lower_band) / basis
    # Chop regime: BB width > 50th percentile of last 50 periods (trending market)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.50).values
    chop_regime = bb_width > bb_width_percentile  # True when trending (width above median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d EMA50 (50) + volume MA (30) + BB (20,50)
    start_idx = 50 + 30 + 20  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(basis[i]) or np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment, and chop regime
            # Long breakout: price breaks above R1 with uptrend, volume regime, and trending market
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_regime[i] and chop_regime[i]
            # Short breakout: price breaks below S1 with downtrend, volume regime, and trending market
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_regime[i] and chop_regime[i]
            
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
            # Exit if price breaks below S1 (mean reversion) or trend changes to downtrend or chop regime ends
            if curr_close < S1_aligned[i] or not uptrend or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above R1 (mean reversion) or trend changes to uptrend or chop regime ends
            if curr_close > R1_aligned[i] or not downtrend or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0