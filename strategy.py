#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray measures bull/bear strength relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d EMA34 provides higher-timeframe trend direction to avoid counter-trend whipsaws
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in bull markets via strong Bull Power + uptrend, bear markets via strong Bear Power + downtrend
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dTrend_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_6h  # Bull Power = High - EMA13
    bear_power = low - ema_13_6h   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 34, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Bull Power > 0 AND price above 1d EMA34 (uptrend)
                if curr_bull_power > 0 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Bear Power < 0 AND price below 1d EMA34 (downtrend)
                elif curr_bear_power < 0 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Bull Power turns negative (bullish momentum fading)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive (bearish momentum fading)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals