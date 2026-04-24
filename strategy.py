#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR-based volatility regime filter and EMA34 trend direction.
- Donchian Channel(20): Upper = 20-period high, Lower = 20-period low.
- Volatility Filter: ATR(14) < 1.5 * ATR(50) to avoid high-vol choppy regimes.
- Trend Filter: Price > EMA34(1d) for long bias, Price < EMA34(1d) for short bias.
- Volume Confirmation: Current volume > 1.3 * 20-period average volume.
- Entry: Long when close > Donchian Upper AND volatility filter AND long bias AND volume confirmation.
         Short when close < Donchian Lower AND volatility filter AND short bias AND volume confirmation.
- Exit: Opposite Donchian break (long exits when close < Donchian Lower, short exits when close > Donchian Upper).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by combining volatility regime filter with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    if len(df_1d) < 50:  # Need sufficient data for ATR(50)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian Channel(20) on 12h timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50)  # Need 20 for Donchian, 50 for ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volatility filter: ATR(14) < 1.5 * ATR(50) to avoid high-vol choppy regimes
        vol_filter = atr14_aligned[i] < 1.5 * atr50_aligned[i]
        
        # Trend filter: price > EMA34 for long bias, price < EMA34 for short bias
        long_bias = curr_close > ema34_1d_aligned[i]
        short_bias = curr_close < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3 * 20-period average volume
        volume_confirm = curr_volume > 1.3 * vol_ma_20_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > highest_high[i]
        breakout_down = curr_close < lowest_low[i]
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close < Donchian Lower
            if position == 1:
                if curr_close < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close > Donchian Upper
            elif position == -1:
                if curr_close > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volatility, trend, and volume filters
        if position == 0:
            # Long: breakout above upper band AND volatility filter AND long bias AND volume confirmation
            long_condition = breakout_up and vol_filter and long_bias and volume_confirm
            
            # Short: breakout below lower band AND volatility filter AND short bias AND volume confirmation
            short_condition = breakout_down and vol_filter and short_bias and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRVolFilter_EMA34Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0