#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla R4/S4 breakouts with 1d EMA50 trend filter and volume spike confirmation.
# Uses weekly Camarilla levels (from prior week) for structural breakout points - R4/S4 represent strong breakout levels.
# 1d EMA50 for trend filter ensures trades align with intermediate-term trend to avoid whipsaws.
# Volume spike (>2x 20-bar average) confirms institutional participation in breakouts.
# Discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Target: 80-160 total trades over 4 years (20-40/year) to balance edge and fee drag.
# Works in both bull/bear: breakouts capture strong moves, trend filter avoids counter-trend, volume confirms validity.

name = "6h_Weekly_Camarilla_R4S4_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter (reduces noise)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for weekly Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels from previous week's OHLC
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Pivot point (PP)
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Camarilla R4 and S4 levels (strong breakout levels)
    r4 = pp + (prev_week_high - prev_week_low) * 1.1 / 2.0 * 2.0  # R4 = PP + 1.1*(H-L)
    s4 = pp - (prev_week_high - prev_week_low) * 1.1 / 2.0 * 2.0  # S4 = PP - 1.1*(H-L)
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 50  # warmup for 1d EMA50
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average (institutional participation)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Weekly Camarilla breakout conditions
        breakout_up = curr_close > r4_aligned[i]   # break above weekly R4
        breakout_down = curr_close < s4_aligned[i] # break below weekly S4
        
        # 1d EMA50 trend filter: price above EMA = uptrend, below = downtrend
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Weekly Camarilla breakout up AND uptrend AND volume spike
            if (breakout_up and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Weekly Camarilla breakout down AND downtrend AND volume spike
            elif (breakout_down and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters weekly Camarilla range OR trend changes to downtrend
            elif (curr_close < r4_aligned[i] and curr_close > s4_aligned[i]) or \
                 not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters weekly Camarilla range OR trend changes to uptrend
            elif (curr_close < r4_aligned[i] and curr_close > s4_aligned[i]) or \
                 not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals