#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter.
- Entry: Long when price breaks above Camarilla R3 AND volume > 1.5 * volume_ma20 AND close > 1d EMA50.
         Short when price breaks below Camarilla S3 AND volume > 1.5 * volume_ma20 AND close < 1d EMA50.
- Exit: Opposite breakout (price < Camarilla S3 for longs, price > Camarilla R3 for shorts) OR close crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels provide institutional support/resistance. Volume spike confirms institutional participation.
- EMA50 filter ensures trading with the daily trend, reducing whipsaws in choppy markets.
- Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on breakout frequency with filters.
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
    """Calculate Camarilla pivot levels for the day."""
    # Typical price
    typical = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    PP = typical
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    
    return R3, S3  # We only need R3 and S3 for breakouts

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
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period volume SMA
    vol_ma20 = sma(volume, 20)
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Calculate Camarilla levels from 1d data
    # We need to calculate R3/S3 for each 1d bar, then align to 12h
    df_1d = df_1d.copy()  # Ensure we have a copy to avoid warnings
    df_1d['R3'], df_1d['S3'] = calculate_camarilla(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['R3'].values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['S3'].values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite breakout OR close crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below S3 OR close falls below 1d EMA50
            if position == 1:
                if curr_close < s3_1d_aligned[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R3 OR close rises above 1d EMA50
            elif position == -1:
                if curr_close > r3_1d_aligned[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: breakout with volume spike and trend filter
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND bullish 1d trend
            if (curr_close > r3_1d_aligned[i] and 
                volume_spike[i] and 
                curr_close > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume spike AND bearish 1d trend
            elif (curr_close < s3_1d_aligned[i] and 
                  volume_spike[i] and 
                  curr_close < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0