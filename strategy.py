#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves.
With 12h EMA34 trend filter and volume confirmation, this strategy works in both bull and bear markets by
only taking breakouts in the direction of the higher timeframe trend.
Designed for low trade frequency (20-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h close for trend
    ema_34_12h = calculate_ema(df_12h['close'].values, 34)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR(20) on 4h for volatility stop
    atr_20 = calculate_atr(high, low, close, 20)
    
    # Calculate Donchian(20) channels on 4h
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_ma.values
    donchian_low = low_ma.values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, EMA, ATR, volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(atr_20[i]) or np.isnan(vol_ma.iloc[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_12h_aligned[i]
        atr_val = atr_20[i]
        vol_spike = volume_spike.iloc[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND price > 12h EMA34 (uptrend)
            long_entry = (curr_close > donchian_high[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Donchian low AND volume spike AND price < 12h EMA34 (downtrend)
            short_entry = (curr_close < donchian_low[i]) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: ATR-based stoploss OR price closes below Donchian low (breakdown)
            stop_price = entry_price - 2.5 * atr_val
            if curr_close < stop_price or curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: ATR-based stoploss OR price closes above Donchian high (breakout)
            stop_price = entry_price + 2.5 * atr_val
            if curr_close > stop_price or curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA34_Trend_VolumeSpike_ATRstop"
timeframe = "4h"
leverage = 1.0