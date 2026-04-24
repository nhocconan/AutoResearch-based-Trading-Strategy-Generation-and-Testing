#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volume spike filter and 1w EMA50 trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR-based volume spike filter (volatility expansion) and 1w EMA50 for trend direction.
- Donchian breakout: Long when price > 20-period high, Short when price < 20-period low.
- Volume confirmation: Current 12h volume > 2.0 * 20-period average volume (calculated on 1d timeframe).
- Trend filter: Price must be above 1w EMA50 for longs, below for shorts.
- Exit: Opposite Donchian breakout OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian breakouts capture volatility expansion moves, effective in both trending and ranging markets.
- 1d ATR volume spike filter ensures breakouts have participation, reducing false signals.
- 1w EMA50 provides strong long-term trend filter to avoid counter-trend trades during major moves.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period=14):
    """Calculate Average True Range."""
    high_low = pd.Series(high - low)
    high_close = pd.Series(np.abs(high - np.roll(close, 1)))
    low_close = pd.Series(np.abs(low - np.roll(close, 1)))
    high_close.iloc[0] = high_low.iloc[0]  # First value
    low_close.iloc[0] = high_low.iloc[0]   # First value
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_values = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_values

def generate_signals(prices):
    n = len(prices)
    if n < 300:  # Need sufficient data for 1w EMA50 (approx 300 bars for 12h data)
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need sufficient data for ATR and volume MA
        return np.zeros(n)
    
    atr_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    
    # Align 1d indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d, additional_delay_bars=1)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 300  # Need sufficient data for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: price < 20-period low OR price falls below 1w EMA50
            if position == 1:
                if curr_low < lowest_low[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > 20-period high OR price rises above 1w EMA50
            elif position == -1:
                if curr_high > highest_high[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Donchian breakout conditions
            bullish_breakout = curr_high > highest_high[i]  # Price breaks above 20-period high
            bearish_breakout = curr_low < lowest_low[i]     # Price breaks below 20-period low
            
            # Volume confirmation: current 12h volume > 2.0 * 20-period average volume (from 1d)
            volume_confirmed = curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            # Long: Bullish Donchian breakout AND price > 1w EMA50 AND volume confirmation
            if bullish_breakout and curr_close > ema50_1w_aligned[i] and volume_confirmed:
                signals[i] = 0.30
                position = 1
            # Short: Bearish Donchian breakout AND price < 1w EMA50 AND volume confirmation
            elif bearish_breakout and curr_close < ema50_1w_aligned[i] and volume_confirmed:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_VolumeSpike_1wEMA50_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0