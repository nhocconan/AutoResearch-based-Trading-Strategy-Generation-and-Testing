#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for trend filter (price above/below EMA50) and 1d for volume regime (above average volume).
- Entry: Long when price breaks above Camarilla R1 AND 4h EMA50 uptrend AND 1d volume > 20-period SMA.
         Short when price breaks below Camarilla S1 AND 4h EMA50 downtrend AND 1d volume > 20-period SMA.
- Exit: Opposite Camarilla break (price crosses R1/S1 in opposite direction) OR 4h EMA50 trend reversal.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Camarilla pivot points identify intraday support/resistance levels that work well in ranging markets.
- EMA50 trend filter ensures we trade with the intermediate-term trend.
- Volume confirmation reduces false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Session filter: 08-20 UTC to avoid low-liquidity Asian session noise.
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema50_4h = ema(df_4h['close'].values, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d volume filter: volume > 20-period SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_sma_1d = sma(df_1d['volume'].values, 20)
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # Calculate Camarilla pivot points on 1h (using previous bar's OHLC)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We need to shift by 1 to avoid look-ahead (use previous bar to calculate levels for current bar)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first bar to NaN since we don't have previous bar
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need sufficient data for EMA/SMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_sma_1d_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla break OR 4h EMA50 trend reversal
        if position != 0:
            # Exit long: price breaks below S1 OR 4h EMA50 turns downtrend
            if position == 1:
                if curr_close < s1[i] or close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R1 OR 4h EMA50 turns uptrend
            elif position == -1:
                if curr_close > r1[i] or close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla break with trend and volume confirmation
        if position == 0 and in_session:
            # Long: price breaks above R1 AND 4h EMA50 uptrend AND volume confirmation
            if curr_close > r1[i] and close[i] > ema50_4h_aligned[i] and curr_volume > vol_sma_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND 4h EMA50 downtrend AND volume confirmation
            elif curr_close < s1[i] and close[i] < ema50_4h_aligned[i] and curr_volume > vol_sma_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dVolumeFilter_v1"
timeframe = "1h"
leverage = 1.0