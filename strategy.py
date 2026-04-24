#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and 1d ATR volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend filter, 1d for ATR volume spike.
- Entry: Long when price breaks above Camarilla H3 AND ATR ratio > 2.0 AND price > 4h EMA34.
         Short when price breaks below Camarilla L3 AND ATR ratio > 2.0 AND price < 4h EMA34.
- Exit: Opposite Camarilla breakout (L3 for longs, H3 for shorts) OR price crosses 4h EMA34 in opposite direction.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 4h EMA34 provides trend filter to avoid counter-trend trades.
- Session filter: 08:00-20:00 UTC to avoid low-volume Asian session noise.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Camarilla levels calculated from prior 1h bar's range to avoid look-ahead.
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

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (H3, L3) from prior bar."""
    range_val = high - low
    h3 = close + range_val * 1.1 / 4
    l3 = close - range_val * 1.1 / 4
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    ema34_4h = ema(df_4h['close'].values, 34)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1h bar (avoid look-ahead)
    # We need to shift by 1 to use prior bar's data
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    # First bar will have invalid shifted values, but we'll handle with min_periods logic
    camarilla_h3, camarilla_l3 = camarilla_levels(high_shift, low_shift, close_shift)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 4h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 4h EMA34
            if position == 1:
                if curr_close < camarilla_l3[i] or curr_close < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 4h EMA34
            elif position == -1:
                if curr_close > camarilla_h3[i] or curr_close > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 2.0 AND bullish 4h trend
            if curr_close > camarilla_h3[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 2.0 AND bearish 4h trend
            elif curr_close < camarilla_l3[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0