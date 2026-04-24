#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA200 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and ATR-based volume spike.
- Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Entry Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA200 AND volume > 1.5 * 20-period average volume.
- Entry Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA200 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Elder Ray signal OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Elder Ray captures momentum strength via price relative to EMA, effective in both bull (buy strength) and bear (sell weakness) markets.
- Volume spike filter ensures participation, reducing false signals.
- 1d EMA200 provides robust trend filter to avoid counter-trend trades.
- Estimated trades: ~100 total over 4 years (~25/year) based on momentum shifts with volume confirmation.
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
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for spike filter
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 1d EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_ma_20_val = vol_ma_20_aligned[i]
        
        # Volume spike condition: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma_20_val if vol_ma_20_val > 0 else False
        
        # Exit conditions: opposite Elder Ray signal OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: Bear Power becomes positive (bearish momentum) OR price falls below 1d EMA200
            if position == 1:
                if bear_power[i] > 0 or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power becomes negative (bullish momentum) OR price rises above 1d EMA200
            elif position == -1:
                if bull_power[i] < 0 or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray momentum with trend filter and volume confirmation
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA200 AND volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and curr_close > ema200_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA200 AND volume spike
            elif bull_power[i] < 0 and bear_power[i] > 0 and curr_close < ema200_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA200_TrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0