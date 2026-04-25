#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_V2
Hypothesis: Reduce trade frequency by tightening Camarilla breakout conditions.
Only enter on breakouts with strong volume confirmation (>1.5x 20-period average volume).
Use 1d EMA34 for trend filter and 1d Choppiness Index for regime adaptation.
Target: 20-40 trades/year by requiring volume spike + trend alignment + session filter.
Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets
by adapting to regime: mean reversion in chop (CHOP>61.8), trend following in trend (CHOP<38.2).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots and EMA34 (loaded ONCE)
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
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                                  np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop_1d = np.where(range_14 > 0, 100 * np.log10(atr_14_1d * 14 / range_14) / np.log10(14), 50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: 20-period average volume on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Require 1.5x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d EMA34 (34) + 1d ATR14 (14) + 1d HH/LL (14) + vol MA (20)
    start_idx = max(34, 14, 20) + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with trend alignment AND volume confirmation
            # Long breakout: price breaks above R1 with uptrend AND volume spike
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i]
            # Short breakout: price breaks below S1 with downtrend AND volume spike
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i]
            
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
            chop = chop_1d_aligned[i]
            if chop > 61.8:  # Ranging market - mean reversion
                if curr_close < S1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop < 38.2:  # Trending market - trend continuation
                if curr_close < S1_aligned[i] * 0.95 or not uptrend:  # Wider stop in trend
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Transition zone
                if curr_close < S1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            chop = chop_1d_aligned[i]
            if chop > 61.8:  # Ranging market - mean reversion
                if curr_close > R1_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop < 38.2:  # Trending market - trend continuation
                if curr_close > R1_aligned[i] * 1.05 or not downtrend:  # Wider stop in trend
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Transition zone
                if curr_close > R1_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_V2"
timeframe = "4h"
leverage = 1.0