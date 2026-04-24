#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout + 1d EMA34 trend filter + volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for trend filter (price above/below EMA34).
- Entry: Long when price breaks above Camarilla R1 AND volume > 1.5x 20-period average AND price > 1d EMA34.
         Short when price breaks below Camarilla S1 AND volume > 1.5x 20-period average AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout (S1 for longs, R1 for shorts) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance derived from prior day's range.
- Volume spike confirms institutional participation.
- 1d EMA34 filter ensures trading with higher timeframe trend.
- Works in bull markets (buying R1 breaks in uptrend) and bear markets (selling S1 breaks in downtrend).
- Estimated trades: ~120 total over 4 years (~30/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    #            S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low),
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    df_1d = get_htf_data(prices, '1d')
    camarilla_R1 = df_1d['close'] + 0.275 * (df_1d['high'] - df_1d['low'])
    camarilla_S1 = df_1d['close'] - 0.275 * (df_1d['high'] - df_1d['low'])
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1.values, additional_delay_bars=1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1.values, additional_delay_bars=1)
    
    # Volume confirmation: volume > 1.5x 20-period SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA and volume SMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price falls below S1 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < camarilla_S1_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above R1 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > camarilla_R1_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND bullish 1d trend
            if curr_close > camarilla_R1_aligned[i] and volume_spike[i] and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND bearish 1d trend
            elif curr_close < camarilla_S1_aligned[i] and volume_spike[i] and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0