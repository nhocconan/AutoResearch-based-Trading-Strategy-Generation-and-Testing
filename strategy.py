#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR filter + volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR regime filter (high volatility environment) and trend bias via price vs EMA50.
- Entry: Long when price breaks above Donchian upper (20) AND 1d ATR(14) > 1d ATR(50) AND price > 1d EMA50.
         Short when price breaks below Donchian lower (20) AND 1d ATR(14) > 1d ATR(50) AND price < 1d EMA50.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels identify volatility breakouts with clear structure.
- 1d ATR ratio filter ensures we only trade in expanding volatility regimes (avoids chop).
- 1d EMA50 provides higher timeframe trend bias to align with dominant momentum.
- Works in bull markets (buy breakouts with uptrend bias) and bear markets (sell breakdowns with downtrend bias).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h timeframe data for volume spike (more responsive than 4h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Volume spike: current volume > 2.0 * 20-period MA of volume
    vol_ma_20 = pd.Series(df_1h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_20, additional_delay_bars=1)
    volume_spike = volume > (2.0 * vol_ma_20_aligned)
    
    # Calculate 1d HTF data for ATR filter and trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d ATR filter: ATR(14) > ATR(50) indicates expanding volatility
    atr_14 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_50 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    atr_ratio = atr_14 / np.where(atr_50 == 0, 1, atr_50)  # Avoid division by zero
    atr_expanding = atr_ratio > 1.0
    atr_expanding_aligned = align_htf_to_ltf(prices, df_1d, atr_expanding.astype(float), additional_delay_bars=1)
    
    # 1d EMA50 for trend bias
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Donchian channels on 4h (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_expanding_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_atr_expanding = atr_expanding_aligned[i] > 0.5  # Convert back to boolean
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1d EMA50
            if position == 1:
                if curr_low <= donchian_low[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1d EMA50
            elif position == -1:
                if curr_high >= donchian_high[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + volume spike + ATR expanding + EMA50 trend alignment
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND ATR expanding AND price > 1d EMA50
            if (curr_high >= donchian_high[i] and 
                curr_volume_spike and 
                curr_atr_expanding and 
                curr_close > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND ATR expanding AND price < 1d EMA50
            elif (curr_low <= donchian_low[i] and 
                  curr_volume_spike and 
                  curr_atr_expanding and 
                  curr_close < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dATR_VolumeSpike_EMA50_TrendBias_v1"
timeframe = "4h"
leverage = 1.0