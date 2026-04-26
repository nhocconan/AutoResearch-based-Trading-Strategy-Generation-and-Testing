#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop_v1
Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA50 trend filter, volume spike confirmation, and ATR-based stoploss.
Targets BTC and ETH for robustness across bull/bear regimes. Donchian breakouts capture medium-term swings,
EMA50 on 12h filters counter-trend noise, volume spike confirms institutional interest, and ATR stop manages risk.
Designed for low trade frequency (target 20-50/year) to minimize fee drag while maintaining edge in ranging and trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume SMA for volume spike detection
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_sma)  # Volume at least 2x average
    
    # Calculate Donchian channels (20-period) on 4h
    # Upper channel = highest high of last 20 periods
    # Lower channel = lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 12h EMA(50), ATR(14), Donchian(20), volume SMA(20)
    start_idx = max(50, 14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filters
        trend_12h_up = close_val > ema_50_12h_aligned[i]
        trend_12h_down = close_val < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper channel AND 12h trend up AND volume spike
            long_signal = (high_val > highest_high[i]) and trend_12h_up and vol_spike
            
            # Short: price breaks below Donchian lower channel AND 12h trend down AND volume spike
            short_signal = (low_val < lowest_low[i]) and trend_12h_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_12h_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_12h_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0