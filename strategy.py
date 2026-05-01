#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and 1d volume confirmation.
# Uses 1w high/low for trend direction (price > weekly midpoint = bullish bias for longs),
# Donchian(20) on 6h for breakout entries, and 1d volume > 1.5x 20-period average for momentum confirmation.
# Session filter (08-20 UTC) reduces noise. ATR-based stoploss (2.0x) manages risk.
# Target: 12-25 trades/year by using weekly trend + Donchian breakouts for high-conviction signals.
# Weekly pivot provides structural bias that works in bull (continuation) and bear (mean reversion from extremes).

name = "6h_Donchian20_1wPivotDir_1dVolConfirm_ATRStop_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly bias: price above pivot = bullish, below = bearish
    weekly_bias = weekly_pivot  # we'll compare price to this
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d data
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian(20) on 6h data
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate ATR(14) for 6h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, donchian_period)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_vol_ma = vol_ma_20_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        
        # Skip if any critical value is NaN
        if (np.isnan(curr_donchian_high) or np.isnan(curr_donchian_low) or 
            np.isnan(curr_weekly_pivot) or np.isnan(curr_vol_ma) or np.isnan(curr_atr)):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x 1d 20-period average
        volume_confirm = curr_volume > (1.5 * curr_vol_ma) if not np.isnan(curr_vol_ma) else False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian(20) high, price above weekly pivot, volume confirmation, in session
            if (curr_close > curr_donchian_high and 
                curr_close > curr_weekly_pivot and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian(20) low, price below weekly pivot, volume confirmation, in session
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_weekly_pivot and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian(20) low OR stoploss hit
            if (curr_close < curr_donchian_low or 
                curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian(20) high OR stoploss hit
            if (curr_close > curr_donchian_high or 
                curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals