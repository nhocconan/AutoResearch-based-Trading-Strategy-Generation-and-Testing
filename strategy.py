#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA200 trend filter and 1d volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and volume spike filter.
- Entry: Long when price breaks above Camarilla H3 AND volume > 1.5x 20-period average volume AND price > 1d EMA200.
         Short when price breaks below Camarilla L3 AND volume > 1.5x 20-period average volume AND price < 1d EMA200.
- Exit: Opposite Camarilla breakout OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades in ranging markets.
- Camarilla levels derived from prior 12h OHLC provide institutional support/resistance.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (H3, L3)."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    h3 = pivot + range_hl * 1.1 / 2.0
    l3 = pivot - range_hl * 1.1 / 2.0
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d volume spike filter: current volume / 20-period average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (avg_vol_20 + 1e-10)  # Avoid division by zero
    
    # Camarilla levels from prior 12h OHLC
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Use prior 12h period's OHLC for current Camarilla levels (no look-ahead)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    # Map 12h indices to 15m indices (12h = 48 * 15m bars)
    bars_per_12h = 48
    
    for i in range(bars_per_12h, n, bars_per_12h):
        # Prior 12h period's OHLC
        start_idx = i - bars_per_12h
        end_idx = i
        if start_idx >= 0 and end_idx <= len(high):
            ph = np.max(high[start_idx:end_idx])
            pl = np.min(low[start_idx:end_idx])
            pc = close[end_idx-1]  # Close of prior 12h period
            h3, l3 = camarilla_levels(ph, pl, pc)
            camarilla_h3[i:end_idx+bars_per_12h] = h3
            camarilla_l3[i:end_idx+bars_per_12h] = l3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, bars_per_12h)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_vol_ratio = vol_ratio[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 1d EMA200
            if position == 1:
                if curr_close < camarilla_l3[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 1d EMA200
            elif position == -1:
                if curr_close > camarilla_h3[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND volume > 1.5x average AND bullish 1d trend
            if curr_close > camarilla_h3[i] and curr_vol_ratio > 1.5 and curr_close > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND volume > 1.5x average AND bearish 1d trend
            elif curr_close < camarilla_l3[i] and curr_vol_ratio > 1.5 and curr_close < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA200_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0