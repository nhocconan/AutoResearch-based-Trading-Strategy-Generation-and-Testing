#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter and 1d for volume average.
- Camarilla levels: calculates H3/L3 from previous 1d bar (high, low, close).
- Entry: Long when price breaks above H3 AND price > 1w EMA34 AND volume > 1.5 * 20-period average volume (1d).
         Short when price breaks below L3 AND price < 1w EMA34 AND volume > 1.5 * 20-period average volume (1d).
- Exit: Opposite Camarilla breakout (H3/L3) or price returns to 1d close (mean reversion to pivot).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Camarilla breakouts work in both bull and bear markets as they capture institutional order flow levels.
- Volume confirmation ensures breakout legitimacy and reduces false signals.
- 1w EMA34 provides robust trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla_levels(high, low, close):
    """
    Calculate Camarilla pivot levels for the next period.
    Based on previous period's high, low, close.
    Returns H3, L3, H4, L4 levels.
    """
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    
    H3 = close + (range_val * 1.1 / 4)
    L3 = close - (range_val * 1.1 / 4)
    H4 = close + (range_val * 1.1 / 2)
    L4 = close - (range_val * 1.1 / 2)
    
    return H3, L3, H4, L4

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d volume average for confirmation (20-period MA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Camarilla levels from 1d data (using previous bar's HLC)
    # We need to shift by 1 to use previous day's data for today's levels
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's HLC to calculate today's Camarilla levels
        phigh = high[i-1] if not np.isnan(high[i-1]) else high[i]
        plow = low[i-1] if not np.isnan(low[i-1]) else low[i]
        pclose = close[i-1] if not np.isnan(close[i-1]) else close[i]
        
        H3, L3, _, _ = calculate_camarilla_levels(phigh, plow, pclose)
        camarilla_H3[i] = H3
        camarilla_L3[i] = L3
    
    # For first bar, use same day's data (will be overridden when we have previous bar)
    if n > 0:
        H3, L3, _, _ = calculate_camarilla_levels(high[0], low[0], close[0])
        camarilla_H3[0] = H3
        camarilla_L3[0] = L3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need 34 for 1w EMA, 20 for volume MA, 1 for Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below L3 OR price returns to 1d close (mean reversion)
            if position == 1:
                if curr_close < camarilla_L3[i] or abs(curr_close - close[i-1]) < 0.001 * close[i-1]:  # Near previous close
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR price returns to 1d close (mean reversion)
            elif position == -1:
                if curr_close > camarilla_H3[i] or abs(curr_close - close[i-1]) < 0.001 * close[i-1]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_close > camarilla_H3[i] and prev_close <= camarilla_H3[i-1]
            breakout_down = curr_close < camarilla_L3[i] and prev_close >= camarilla_L3[i-1]
            
            # Trend filter: price vs 1w EMA34
            long_trend = curr_close > ema34_1w_aligned[i]
            short_trend = curr_close < ema34_1w_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if breakout_up and long_trend and volume_confirm:
                signals[i] = 0.30
                position = 1
            elif breakout_down and short_trend and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0