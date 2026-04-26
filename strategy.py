#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ChopRegime
Hypothesis: Trade 4h Donchian(20) breakouts with 12h EMA50 trend filter, volume confirmation, and chop regime filter.
Targets 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.
Donchian breakouts capture momentum bursts; 12h EMA50 ensures trading with intermediate trend.
Volume confirmation adds conviction to breakouts. Chop regime filter (CHOP < 38.2) ensures trending market.
Works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter via opposite breakouts).
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels from 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Choppiness regime filter: CHOP(14) < 38.2 = trending market (good for breakouts)
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum()
    true_range = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(hl_range / true_range) / np.log10(14)
    chop_regime = chop < 38.2  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of 12h EMA(50), Donchian(20), volume MA(20), ATR(14), CHOP(14)
    start_idx = max(50, 20, 20, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_12h_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_12h_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        regime_ok = chop_regime[i]  # trending market regime
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND volume confirm AND 12h trend up AND trending regime
            long_signal = (close_val > highest_20[i]) and vol_conf and trend_12h_up and regime_ok
            
            # Short: price breaks below Donchian lower band AND volume confirm AND 12h trend down AND trending regime
            short_signal = (close_val < lowest_20[i]) and vol_conf and trend_12h_down and regime_ok
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # ATR trailing stop: exit if price drops 2.0 * ATR from highest since entry
            if close_val < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips down or regime changes to ranging
            elif not trend_12h_up or not regime_ok:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.0 * ATR from lowest since entry
            if close_val > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips up or regime changes to ranging
            elif not trend_12h_down or not regime_ok:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0