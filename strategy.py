#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR-based regime filter (trend vs range) and 1w for EMA50 trend direction.
- Donchian breakout: Long when price > 20-period high AND ATR(14) < ATR(50) (low volatility regime) AND volume > 1.5 * 20-period average volume.
         Short when price < 20-period low AND ATR(14) < ATR(50) AND volume > 1.5 * 20-period average volume.
- ATR regime: Only trade when short-term ATR < long-term ATR (low volatility, breakout-prone conditions).
- Exit: Opposite Donchian breakout (price < 10-period high for long exit, price > 10-period low for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets (strong breakouts) and bear markets (strong breakdowns) with volatility filter avoiding false breakouts in high volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    ranges = np.concatenate([high_low[:, None], high_close[:, None], low_close[:, None]], axis=1)
    tr = np.max(ranges, axis=1)
    tr[0] = high_low[0]  # First TR is just high-low
    atr_vals = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_vals

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR regime filter (ATR14 < ATR50 = low volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR50
        return np.zeros(n)
    
    atr14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr50_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    atr_regime_1d = atr14_1d < atr50_1d  # True when ATR14 < ATR50 (low volatility)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_1d.astype(float))
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian high/low: 20-period rolling max/min
    donch_high_12h = pd.Series(df_12h['close'].values).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(df_12h['close'].values).rolling(window=20, min_periods=20).min().values
    
    # Donchian exit channels: 10-period rolling max/min
    donch_exit_high_12h = pd.Series(df_12h['close'].values).rolling(window=10, min_periods=10).max().values
    donch_exit_low_12h = pd.Series(df_12h['close'].values).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    donch_exit_high_aligned = align_htf_to_ltf(prices, df_12h, donch_exit_high_12h)
    donch_exit_low_aligned = align_htf_to_ltf(prices, df_12h, donch_exit_low_12h)
    
    # Calculate 12h volume average for confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_exit_high_aligned[i]) or np.isnan(donch_exit_low_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_regime_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout (10-period channels)
        if position != 0:
            # Exit long: price < 10-period high
            if position == 1:
                if curr_close < donch_exit_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > 10-period low
            elif position == -1:
                if curr_close > donch_exit_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with ATR regime and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
            
            # ATR regime filter: only trade in low volatility (ATR14 < ATR50)
            atr_regime = bool(atr_regime_aligned[i] > 0.5)  # Convert to boolean
            
            # Trend filter: price > 1w EMA50 for long, price < 1w EMA50 for short
            trend_filter_long = curr_close > ema50_1w_aligned[i]
            trend_filter_short = curr_close < ema50_1w_aligned[i]
            
            # Long: price > 20-period high AND ATR regime AND volume confirm AND trend filter
            long_condition = (curr_close > donch_high_aligned[i] and
                            atr_regime and
                            volume_confirm and
                            trend_filter_long)
            
            # Short: price < 20-period low AND ATR regime AND volume confirm AND trend filter
            short_condition = (curr_close < donch_low_aligned[i] and
                             atr_regime and
                             volume_confirm and
                             trend_filter_short)
            
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

name = "12h_Donchian20_Breakout_1dATRRegime_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0