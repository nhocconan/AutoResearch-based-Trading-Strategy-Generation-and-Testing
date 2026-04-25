#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeConfirm
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar avg).
Targets 20-50 trades/year by requiring: 1) price breaks Donchian channel, 2) aligned with 1d EMA trend, 3) volume spike.
Designed for low turnover and high edge via trend alignment and institutional volume confirmation. Uses ATR(14) stoploss.
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
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d data for higher timeframe trend (optional: 1d EMA200 for regime)
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian Channel (20) on 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) and 1d EMA50 (50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend and regime filters
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        bull_regime = curr_close > ema_200_1d_aligned[i]  # bull market filter
        bear_regime = curr_close < ema_200_1d_aligned[i]  # bear market filter
        
        if position == 0:
            # Look for entry signals with volume confirmation
            # Long breakout: price breaks above Donchian upper with uptrend and volume confirmation
            long_breakout = (curr_close > highest_20[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below Donchian lower with downtrend and volume confirmation
            short_breakout = (curr_close < lowest_20[i]) and downtrend and volume_confirm[i]
            
            # In bull regime, favor longs; in bear regime, favor shorts
            if long_breakout and (bull_regime or not bear_regime):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout and (bear_regime or not bull_regime):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below Donchian lower (mean reversion) or trend changes
            elif curr_close < lowest_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above Donchian upper (mean reversion) or trend changes
            elif curr_close > highest_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0