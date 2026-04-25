#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA50 trend filter, volume confirmation, and choppiness regime filter.
Targets 20-30 trades/year by requiring: 1) price breaks 20-period Donchian channel, 2) aligned with 1d EMA50 trend,
3) volume > 1.8x 20-period average, 4) choppy market filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend).
Uses 4h timeframe to capture significant moves with controlled frequency. Volume spike reduces false breakouts.
Choppiness filter ensures we only trend-follow in strong trends and mean-revert in ranging markets, improving performance
in both bull and bear markets by adapting to market regime.
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
    
    # 1d data for choppiness regime filter (loaded ONCE)
    # Chop = 100 * log10(sum(ATR(14),14) / (log10(HH(14)-LL(14)) * sqrt(14)))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    sum_atr14 = atr14.rolling(window=14, min_periods=14).sum()
    hh14 = df_1d['high'].rolling(window=14, min_periods=14).max()
    ll14 = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * (np.log10(sum_atr14) - np.log10((hh14 - ll14) * np.sqrt(14))) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Donchian channel (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA50 (50) + chop (14+14=28) + Donchian (20) + volume MA (20)
    start_idx = 60  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Regime filter: choppy vs trending
        chop_value = chop_aligned[i]
        is_choppy = chop_value > 61.8  # ranging market -> mean reversion
        is_trending = chop_value < 38.2  # trending market -> trend follow
        
        if position == 0:
            # Look for entry signals with volume confirmation
            if is_trending:
                # In trending markets: Donchian breakout with trend
                long_breakout = (curr_close > donchian_high[i]) and uptrend and volume_confirm[i]
                short_breakout = (curr_close < donchian_low[i]) and downtrend and volume_confirm[i]
            elif is_choppy:
                # In choppy markets: mean reversion at Donchian edges
                long_breakout = (curr_close < donchian_low[i]) and volume_confirm[i]  # bounce from lower band
                short_breakout = (curr_close > donchian_high[i]) and volume_confirm[i]  # rejection from upper band
            else:
                # Transition regime: no entries
                long_breakout = False
                short_breakout = False
            
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
            # Exit if: trend reverses, or chop regime changes and price hits opposite band, or time-based exit
            if not uptrend or (is_choppy and curr_close > donchian_high[i]) or (not is_trending and not is_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if: trend reverses, or chop regime changes and price hits opposite band, or time-based exit
            if not downtrend or (is_choppy and curr_close < donchian_low[i]) or (not is_trending and not is_choppy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0