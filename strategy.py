#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d EMA200 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA200 trend filter (long-term trend) and ATR-based volume spike detection.
- Donchian(20): Upper = 20-period high, Lower = 20-period low.
- Entry: Long when close > Upper Band AND price > 1d EMA200 AND volume > 2.0 * 20-period ATR.
         Short when close < Lower Band AND price < 1d EMA200 AND volume > 2.0 * 20-period ATR.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian breakouts capture momentum in trending markets and reversals in ranging markets.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades in bear markets.
- Volume confirmation via ATR spike ensures breakouts have conviction, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high[0] - low[0]  # First TR
    return tr.ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR(20) for volume confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    atr20_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr20_1d, additional_delay_bars=1)
    
    # Calculate Donchian(20) channels
    # Upper Band: 20-period high
    upper_raw = pd.Series(close).rolling(window=20, min_periods=20).max().values
    # Lower Band: 20-period low
    lower_raw = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr20_1d_aligned[i]) or
            np.isnan(upper_raw[i]) or np.isnan(lower_raw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: price < Lower Band OR price falls below 1d EMA200
            if position == 1:
                if curr_close < lower_raw[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > Upper Band OR price rises above 1d EMA200
            elif position == -1:
                if curr_close > upper_raw[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * ATR20
            volume_confirm = curr_volume > (2.0 * atr20_1d_aligned[i])
            
            # Long: break above Upper Band AND price > 1d EMA200 AND volume confirmation
            if curr_close > upper_raw[i] and curr_close > ema200_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below Lower Band AND price < 1d EMA200 AND volume confirmation
            elif curr_close < lower_raw[i] and curr_close < ema200_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA200_TrendFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0