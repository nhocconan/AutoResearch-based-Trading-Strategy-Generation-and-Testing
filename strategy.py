#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout + 1d volume spike + 12h EMA50 trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume confirmation and 12h for EMA trend filter.
- Entry: Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average AND price > 12h EMA50.
         Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average AND price < 12h EMA50.
- Exit: Opposite Camarilla breakout OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance based on prior day's range.
- Volume spike confirms institutional participation in breakout.
- 12h EMA50 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_levels(high, low, close):
    """
    Calculate Camarilla pivot levels for intraday trading.
    Based on previous day's high, low, close.
    Returns: (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    """
    range_val = high - low
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    pp = (high + low + close) / 3
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r4, r3, r2, r1, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 1d volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / (vol_ma20 + 1e-10)  # Avoid division by zero
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d data
    # We need previous day's high, low, close for today's levels
    r1_1d = np.full(len(df_1d), np.nan)
    s1_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's data to calculate today's Camarilla levels
        _, _, _, r1, _, s1, _, _, _ = camarilla_levels(
            df_1d['high'].values[i-1],
            df_1d['low'].values[i-1],
            df_1d['close'].values[i-1]
        )
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d, additional_delay_bars=1)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below S1 OR price falls below 12h EMA50
            if position == 1:
                if curr_close < s1_1d_aligned[i] or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R1 OR price rises above 12h EMA50
            elif position == -1:
                if curr_close > r1_1d_aligned[i] or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above R1 AND volume > 1.5x average AND bullish 12h trend
            if curr_close > r1_1d_aligned[i] and vol_ratio_aligned[i] > 1.5 and curr_close > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 1.5x average AND bearish 12h trend
            elif curr_close < s1_1d_aligned[i] and vol_ratio_aligned[i] > 1.5 and curr_close < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dVolumeSpike_12hEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0