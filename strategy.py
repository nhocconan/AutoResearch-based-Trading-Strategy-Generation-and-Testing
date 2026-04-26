#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Donchian(20) breakouts aligned with 12h EMA50 trend and volume confirmation (>2x average) capture strong momentum moves while avoiding counter-trend whipsaws. Uses ATR(14) stoploss to manage risk. Designed for 4h to target 20-50 trades/year with discrete sizing (0.25). Works in bull/bear via 12h trend filter.
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 24*4h = 4d)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA(50), Donchian(20), volume(24), ATR(14)
    start_idx = max(50, 20, 24, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_12h_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(upper) or 
            np.isnan(lower) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Trend filter: price vs 12h EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Breakout conditions
        bull_breakout = close_val > upper
        bear_breakout = close_val < lower
        
        # Long: price breaks above upper Donchian with 12h uptrend and volume
        long_condition = bull_breakout and uptrend and volume_confirmed
        # Short: price breaks below lower Donchian with 12h downtrend and volume
        short_condition = bear_breakout and downtrend and volume_confirmed
        
        # Stoploss: ATR-based (2.5 * ATR)
        long_stop = position == 1 and close_val < (entry_price - 2.5 * atr_val)
        short_stop = position == -1 and close_val > (entry_price + 2.5 * atr_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_stop:
            signals[i] = 0.0
            position = 0
        elif short_stop:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0