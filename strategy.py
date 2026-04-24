#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian breakout + 1w ATR volume spike + 1w EMA trend filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for ATR volume spike and EMA trend filter.
- Entry: Long when price breaks above Donchian(20) high AND 1w ATR ratio > 1.5 AND price > 1w EMA34.
         Short when price breaks below Donchian(20) low AND 1w ATR ratio > 1.5 AND price < 1w EMA34.
- Exit: Opposite Donchian breakout OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels identify volatility breakouts.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion.
- 1w EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~50 total over 4 years (~12/year) based on volatility breakout frequency with filters.
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

def donchian_channels(high, low, period):
    """Calculate Donchian Channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Calculate 1w ATR for volume spike filter
    if len(df_1w) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 20)
    atr_current = atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio, additional_delay_bars=1)
    
    # Donchian channels on 1d (20-period)
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1w EMA34
            if position == 1:
                if curr_close < donch_lo[i] or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > donch_hi[i] or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Donchian high AND ATR ratio > 1.5 AND bullish 1w trend
            if curr_close > donch_hi[i] and atr_ratio_aligned[i] > 1.5 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND ATR ratio > 1.5 AND bearish 1w trend
            elif curr_close < donch_lo[i] and atr_ratio_aligned[i] > 1.5 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_1wATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0