#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3S3 breakout + 1d volume spike + ADX trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend filter and volume confirmation.
- Entry: Long when price breaks above Camarilla R3 level AND 1d ADX > 25 AND 1d volume > 1.5 * 20-period average volume.
         Short when price breaks below Camarilla S3 level AND 1d ADX > 25 AND 1d volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla breakout (R4/S4) OR price crosses 1d VWAP in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide high-probability reversal/breakout levels from institutional order flow.
- ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.
- Volume spike confirms institutional participation in the breakout.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with filters.
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

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / (tr_period + 1e-10)
    di_minus = 100 * dm_minus_period / (tr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels."""
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    R4 = pivot + (range_hl * 1.1 / 2)
    R3 = pivot + (range_hl * 1.1 / 4)
    R2 = pivot + (range_hl * 1.1 / 6)
    R1 = pivot + (range_hl * 1.1 / 12)
    
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)
    S4 = pivot - (range_hl * 1.1 / 2)
    
    return R4, R3, R2, R1, pivot, S1, S2, S3, S4

def vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    vwap_values = np.cumsum(typical_price * volume) / (np.cumsum(volume) + 1e-10)
    return vwap_values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h HTF data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h
    camarilla_R4, camarilla_R3, camarilla_R2, camarilla_R1, camarilla_pivot, camarilla_S1, camarilla_S2, camarilla_S3, camarilla_S4 = camarilla_levels(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values
    )
    
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R3, additional_delay_bars=1)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S3, additional_delay_bars=1)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R4, additional_delay_bars=1)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S4, additional_delay_bars=1)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    adx_values = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values, additional_delay_bars=1)
    
    # Calculate 1d volume spike filter
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio, additional_delay_bars=1)
    
    # Calculate 1d VWAP for exit
    vwap_1d = vwap(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, df_1d['volume'].values)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or
            np.isnan(vwap_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d VWAP in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla S4 OR price falls below 1d VWAP
            if position == 1:
                if curr_close < camarilla_S4_aligned[i] or curr_close < vwap_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R4 OR price rises above 1d VWAP
            elif position == -1:
                if curr_close > camarilla_R4_aligned[i] or curr_close > vwap_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla R3 AND volume spike AND ADX > 25 (trending)
            if curr_close > camarilla_R3_aligned[i] and vol_ratio_aligned[i] > 1.5 and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND volume spike AND ADX > 25 (trending)
            elif curr_close < camarilla_S3_aligned[i] and vol_ratio_aligned[i] > 1.5 and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike_VWAPExit_v1"
timeframe = "12h"
leverage = 1.0