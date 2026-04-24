#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout + 1d volume spike + ADX trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend filter and volume confirmation, 1w for Camarilla pivot context (optional).
- Entry: Long when price breaks above Camarilla H3 (1d) AND 1d volume > 1.5 * 20-period avg volume AND 1d ADX > 25.
         Short when price breaks below Camarilla L3 (1d) AND 1d volume > 1.5 * 20-period avg volume AND 1d ADX > 25.
- Exit: Opposite Camarilla breakout (H3/L3) OR price crosses Camarilla H4/L4 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide precise intraday support/resistance derived from prior day's range.
- Volume confirmation ensures breakouts have participation.
- ADX > 25 ensures we only trade strong trends, reducing whipsaws in ranging markets.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend).
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

def adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    plus_di = 100 * ema(plus_dm, period) / (ema(true_range, period) + 1e-10)
    minus_di = 100 * ema(minus_dm, period) / (ema(true_range, period) + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_vals = ema(dx, period)
    
    return adx_vals

def camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_ = high - low
    h4 = close + range_ * 1.1 / 2
    h3 = close + range_ * 1.1 / 4
    l3 = close - range_ * 1.1 / 4
    l4 = close - range_ * 1.1 / 2
    return h4, h3, l3, l4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: ADX > 25
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    adx_14 = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14, additional_delay_bars=1)
    
    # Calculate 1d volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d
    h4_1d, h3_1d, l3_1d, l4_1d = camarilla_pivots(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d, additional_delay_bars=1)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d, additional_delay_bars=1)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d, additional_delay_bars=1)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses H4/L4 in opposite direction
        if position != 0:
            # Exit long: price breaks below L3 OR price falls below L4
            if position == 1:
                if curr_close < l3_1d_aligned[i] or curr_close < l4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR price rises above H4
            elif position == -1:
                if curr_close > h3_1d_aligned[i] or curr_close > h4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above H3 AND volume > 1.5 * 20-period avg AND ADX > 25
            if curr_close > h3_1d_aligned[i] and vol_ratio_aligned[i] > 1.5 and adx_14_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume > 1.5 * 20-period avg AND ADX > 25
            elif curr_close < l3_1d_aligned[i] and vol_ratio_aligned[i] > 1.5 and adx_14_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dVolumeSpike_ADXTrendFilter_v1"
timeframe = "12h"
leverage = 1.0