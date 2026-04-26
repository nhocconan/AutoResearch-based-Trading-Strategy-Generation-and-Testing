#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 12h timeframe, trade Donchian(20) breakouts filtered by 1d EMA50 trend and volume confirmation. 
Donchian channels provide clear breakout levels that work in both trending and ranging markets. 
The 1d EMA50 acts as a higher-timeframe trend filter: only take longs when price is above 1d EMA50 (bullish regime) 
and shorts when below 1d EMA50 (bearish regime). Volume spike confirms institutional participation at the breakout. 
Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag. 
Works in bull/bear markets via 1d EMA50 filter and ATR-based stoploss.
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
    
    # Donchian(20) on primary 12h timeframe
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Get 1d data for EMA50 trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # ATR(14) for dynamic stoploss on 12h
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), EMA50, ATR(14), volume MA(20)
    start_idx = max(period, 50, atr_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma_20_val = vol_ma_20_aligned[i]
        atr_val = atr[i]
        volume_val = volume[i]
        
        # Volume spike: current 12h volume > 2.0 * 20-period 1d volume average (aligned)
        volume_spike = volume_val > (2.0 * vol_ma_20_val)
        
        if position == 0:
            # Long: price breaks above Donchian high, above 1d EMA50, volume spike
            long_breakout = close_val > donchian_high_val
            above_ema50 = close_val > ema_50_val
            long_signal = long_breakout and above_ema50 and volume_spike
            
            # Short: price breaks below Donchian low, below 1d EMA50, volume spike
            short_breakout = close_val < donchian_low_val
            below_ema50 = close_val < ema_50_val
            short_signal = short_breakout and below_ema50 and volume_spike
            
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
            # Exit: price breaks below Donchian low OR ATR-based stoploss
            breakout_exit = close_val < donchian_low_val
            stoploss_exit = close_val < entry_price - 2.5 * atr_val
            if breakout_exit or stoploss_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR ATR-based stoploss
            breakout_exit = close_val > donchian_high_val
            stoploss_exit = close_val > entry_price + 2.5 * atr_val
            if breakout_exit or stoploss_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0