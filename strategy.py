#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout + 1d volume confirmation + ATR volatility filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for Donchian(20) breakout direction, 1d for volume spike and ATR regime filter.
- Entry: Long when price breaks above 4h Donchian upper channel AND 1d volume > 1.5 * 20-day average AND 1d ATR(14) < 1d ATR(50) (low volatility regime).
         Short when price breaks below 4h Donchian lower channel AND same volume/volatility conditions.
- Exit: Opposite Donchian breakout OR volatility regime shift (ATR(14) > ATR(50)).
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Donchian breakouts capture sustained moves; volume confirmation ensures participation; low volatility filter avoids choppy markets.
- Works in bull (breakouts with volume) and bear (volatility regime shifts enable short bias in ranging markets).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period=14):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    ranges = np.column_stack([high_low, high_close, low_close])
    true_range = np.max(ranges, axis=1)
    true_range[0] = high_low[0]  # First period
    atr_vals = pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_vals

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:  # Need sufficient data for Donchian
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / (vol_ma_20 + 1e-10)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 1d ATR volatility filter (ATR(14) < ATR(50) = low vol regime)
    atr_14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_50_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    vol_regime = atr_14_1d < atr_50_1d  # True when low volatility regime
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_ratio = vol_ratio_1d_aligned[i]
        curr_vol_regime = vol_regime_aligned[i] > 0.5  # Convert back to boolean
        
        # Exit conditions: opposite Donchian breakout OR volatility regime shift
        if position != 0:
            # Exit long: price breaks below lower channel OR volatility regime shifts to high vol
            if position == 1:
                if curr_close < donchian_lower_aligned[i] or not curr_vol_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper channel OR volatility regime shifts to high vol
            elif position == -1:
                if curr_close > donchian_upper_aligned[i] or not curr_vol_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and low volatility regime
        if position == 0:
            bullish_breakout = curr_close > donchian_upper_aligned[i]
            bearish_breakout = curr_close < donchian_lower_aligned[i]
            volume_confirm = curr_volume_ratio > 1.5
            
            # Long: Bullish breakout AND volume confirmation AND low volatility regime
            if bullish_breakout and volume_confirm and curr_vol_regime:
                signals[i] = 0.20
                position = 1
            # Short: Bearish breakout AND volume confirmation AND low volatility regime
            elif bearish_breakout and volume_confirm and curr_vol_regime:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_Breakout_1dVolumeVolatilityFilter_v1"
timeframe = "1h"
leverage = 1.0