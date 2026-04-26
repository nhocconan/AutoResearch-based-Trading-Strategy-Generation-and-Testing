#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1
Hypothesis: On 4h timeframe, Donchian(20) breakouts filtered by 12h EMA50 trend and volume spike capture strong momentum moves with controlled trade frequency. Long when price breaks above Donchian upper in bullish 12h trend with volume > 1.5x average; short when price breaks below Donchian lower in bearish 12h trend with volume spike. Uses ATR-based stoploss and discrete sizing (±0.25) to target 20-50 trades/year. Works in both bull/bear markets by only trading in direction of higher-timeframe trend.
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for higher-timeframe trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period) on 4h
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Volume average (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0  # first bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian (20), volume MA (20), ATR (14) + 12h EMA50 alignment
    start_idx = max(20, 20, 14) + 4  # +4 to ensure 12h bar completion (4h -> 12h: 3 bars, but add buffer)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_50_val = ema_50_12h_aligned[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_50_val
        bearish_12h = close_val < ema_50_val
        
        # Volume spike: current volume > 1.5x 20-period average
        volume_spike = vol_val > 1.5 * vol_ma_val
        
        # Entry conditions: Donchian breakout in direction of 12h trend with volume spike
        long_entry = (close_val > upper_val) and bullish_12h and volume_spike
        short_entry = (close_val < lower_val) and bearish_12h and volume_spike
        
        # Stoploss conditions: ATR-based stop
        long_stop = position == 1 and close_val < upper_val - 2.0 * atr_val
        short_stop = position == -1 and close_val > lower_val + 2.0 * atr_val
        
        # Exit conditions: breakout failure or stoploss
        long_exit = long_stop or (position == 1 and close_val < upper_val)
        short_exit = short_stop or (position == -1 and close_val > lower_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0