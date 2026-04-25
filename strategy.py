#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 1d EMA34 trend + volume confirmation + ATR stoploss
Hypothesis: Donchian breakouts capture institutional accumulation/distribution zones.
1d EMA34 provides multi-timeframe trend filter to avoid counter-trend trades.
Volume confirmation ensures breakout validity. ATR stoploss manages risk.
Designed for 15-30 trades/year per symbol to minimize fee drag while maintaining edge.
Works in bull markets (breakout continuation) and bear markets (mean reversion to trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Daily data for EMA34 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) - using current bar's high/low for breakout
    # We need to use previous 20 bars to avoid look-ahead
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, volume MA, ATR, and EMA
    start_idx = max(20, 20, 14, 34) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > donchian_high[i]
        breakout_short = curr_close < donchian_low[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + daily EMA34 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Donchian breakdown, trend change, or ATR stoploss
            stoploss_level = entry_price - 2.5 * atr[i]
            if curr_close < donchian_low[i] or curr_close < ema_34_1d_aligned[i] or curr_close < stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Donchian breakout, trend change, or ATR stoploss
            stoploss_level = entry_price + 2.5 * atr[i]
            if curr_close > donchian_high[i] or curr_close > ema_34_1d_aligned[i] or curr_close > stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0