#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend filter (price above/below EMA50).
- Entry: Long when price breaks above Camarilla R1 level AND volume > 1.5x volume SMA20 AND price > 1w EMA50.
         Short when price breaks below Camarilla S1 level AND volume > 1.5x volume SMA20 AND price < 1w EMA50.
- Exit: Opposite Camarilla breakout (price crosses back below R1 for longs, above S1 for shorts) OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide precise intraday support/resistance derived from prior day's range.
- Volume spike confirms institutional participation behind the breakout.
- 1w EMA50 filter ensures alignment with higher-timeframe trend to avoid counter-trend whipsaws.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close  # fallback to close if no range
    camarilla_h4 = close + range_val * 1.1 / 2
    camarilla_h3 = close + range_val * 1.1 / 4
    camarilla_h2 = close + range_val * 1.1 / 6
    camarilla_h1 = close + range_val * 1.1 / 12
    camarilla_l1 = close - range_val * 1.1 / 12
    camarilla_l2 = close - range_val * 1.1 / 6
    camarilla_l3 = close - range_val * 1.1 / 4
    camarilla_l4 = close - range_val * 1.1 / 2
    return camarilla_h3, camarilla_l3, camarilla_h4, camarilla_l4  # H3, L3, H4, L4 (R1=H3, S1=L3)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate volume SMA20 for confirmation
    volume_sma20 = sma(volume, 20)
    
    # Calculate Camarilla levels from 1d data (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior day's OHLC for today's Camarilla levels (no look-ahead)
    camarilla_data = calculate_camarilla(
        df_1d['high'].shift(1).values,  # prior day high
        df_1d['low'].shift(1).values,   # prior day low
        df_1d['close'].shift(1).values  # prior day close
    )
    camarilla_h3_1d = camarilla_data[0]  # H3 = R1
    camarilla_l3_1d = camarilla_data[1]  # L3 = S1
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA/SMA/alignment
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or
            np.isnan(volume_sma20[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla S1 (L3) OR price falls below 1w EMA50
            if position == 1:
                if curr_close < camarilla_l3_aligned[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R1 (H3) OR price rises above 1w EMA50
            elif position == -1:
                if curr_close > camarilla_h3_aligned[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend alignment
        if position == 0:
            volume_confirm = curr_volume > 1.5 * volume_sma20[i]
            
            # Long: price breaks above Camarilla R1 (H3) AND volume confirmation AND bullish 1w trend
            if curr_close > camarilla_h3_aligned[i] and volume_confirm and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 (L3) AND volume confirmation AND bearish 1w trend
            elif curr_close < camarilla_l3_aligned[i] and volume_confirm and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0