#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above 20-period high AND 1d volume > 2.0 * 20-period average volume AND 1d ADX > 25 (trending market).
# Short when price breaks below 20-period low AND 1d volume > 2.0 * 20-period average volume AND 1d ADX > 25.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to balance reward and risk. Designed for BTC/ETH robustness by capturing strong trends with volume confirmation.
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.

name = "4h_Donchian20_Breakout_1dVolumeSpike_ADXTrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d indicators (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX calculation (trend strength filter)
    adx_period = 14
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # 1d volume spike filter
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 4h Donchian channels (primary timeframe)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_period, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC (optional, can be removed if too restrictive)
        hour = pd.Timestamp(prices.iloc[i]['open_time']).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian high AND volume spike AND strong trend (ADX > 25)
            if (open_[i] <= highest_high[i] and close[i] > highest_high[i] and 
                adx_aligned[i] > 25 and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low AND volume spike AND strong trend (ADX > 25)
            elif (open_[i] >= lowest_low[i] and close[i] < lowest_low[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals