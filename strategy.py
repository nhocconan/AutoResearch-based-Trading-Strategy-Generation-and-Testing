#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA50 trend filter, volume confirmation (>1.5x 20-period average),
and choppiness regime filter (CHOP(14) < 61.8 to avoid ranging markets). Targets 20-40 trades/year by requiring
confluence of trend, momentum, volume, and regime conditions. Uses discrete position sizing (0.25) to minimize
fee churn. Designed to work in both bull and bear markets by following the 1d trend direction and avoiding
entries in high-chop regimes where breakouts fail.
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
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels from 1d data (for structure)
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Choppiness Index regime filter: CHOP(14) < 61.8 = trending (favor breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_14 = pd.Series(np.maximum.reduce([
        high[1:] - low[:-1],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[:-1] - close[:-1])
    ]).copy(), index=prices.index).rolling(window=14, min_periods=14).mean().values
    atr_14 = np.concatenate([np.full(14, np.nan), atr_14])  # align length
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14) / np.log10(highest_high_14 - lowest_low_14 + 1e-10)
    chop_regime = chop < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA50 (50) + Donchian(20) (20) + ATR(14) (14) + volume MA (20)
    start_idx = 50 + 20 + 14 + 20  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
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
            # Long breakout: price breaks above 1d Donchian high with uptrend, volume confirmation, and trending regime
            long_breakout = (curr_close > donchian_high_20_aligned[i]) and uptrend and volume_confirm[i] and chop_regime[i]
            # Short breakout: price breaks below 1d Donchian low with downtrend, volume confirmation, and trending regime
            short_breakout = (curr_close < donchian_low_20_aligned[i]) and downtrend and volume_confirm[i] and chop_regime[i]
            
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
            # Exit if price breaks below 1d Donchian low (mean reversion) or trend changes to downtrend or chop regime increases
            if curr_close < donchian_low_20_aligned[i] or not uptrend or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above 1d Donchian high (mean reversion) or trend changes to uptrend or chop regime increases
            if curr_close > donchian_high_20_aligned[i] or not downtrend or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0