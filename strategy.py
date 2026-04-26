#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopRegime
Hypothesis: Trade 12h Donchian(20) breakouts with 1d EMA34 trend filter, volume confirmation, and chop regime filter.
Donchian breakouts capture momentum; EMA34 ensures trading with dominant trend; volume confirms conviction;
Chop regime filter avoids false breakouts in ranging markets. Works in bull (breakouts with trend) and bear
(mean reversion at extremes with trend filter). Targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels from price history
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
    
    # Warmup: max of 1d EMA(34), Donchian(20), volume MA(20), ATR(14), CHOP(14)
    start_idx = max(34, 20, 20, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_ma[i]) or
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
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        regime_ok = chop_regime[i]  # trending market regime
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume confirm AND 1d trend up AND trending regime
            long_signal = (close_val > highest_20[i]) and vol_conf and trend_1d_up and regime_ok
            
            # Short: price breaks below lower Donchian AND volume confirm AND 1d trend down AND trending regime
            short_signal = (close_val < lowest_20[i]) and vol_conf and trend_1d_down and regime_ok
            
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
            # ATR trailing stop: exit if price drops 2.5 * ATR from highest since entry
            if close_val < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips down or regime changes to ranging
            elif not trend_1d_up or not regime_ok:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.5 * ATR from lowest since entry
            if close_val > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips up or regime changes to ranging
            elif not trend_1d_down or not regime_ok:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0