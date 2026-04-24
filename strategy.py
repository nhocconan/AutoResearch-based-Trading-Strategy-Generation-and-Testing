#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for trend filter (price above/below EMA34).
- Entry: Long when price breaks above Camarilla R1 level AND close > 12h EMA34 AND volume > 1.5 * average volume.
         Short when price breaks below Camarilla S1 level AND close < 12h EMA34 AND volume > 1.5 * average volume.
- Exit: Opposite Camarilla breakout OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance based on prior day's range.
- EMA34 filter ensures trades align with medium-term trend.
- Volume confirmation reduces false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~150 total over 4 years (~38/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high, low, close
    prior_high = df_1d['high'].shift(1).values  # Shift to avoid look-ahead
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R1 = prior_close + (prior_high - prior_low) * 1.1 / 12
    S1 = prior_close - (prior_high - prior_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1, additional_delay_bars=1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA/volume/Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below S1 OR price falls below 12h EMA34
            if position == 1:
                if curr_close < S1_aligned[i] or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R1 OR price rises above 12h EMA34
            elif position == -1:
                if curr_close > R1_aligned[i] or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction with volume confirmation
        if position == 0:
            volume_confirmed = curr_volume > 1.5 * avg_volume[i]
            
            # Long: price breaks above R1 AND bullish 12h trend AND volume confirmation
            if curr_close > R1_aligned[i] and close > ema34_12h_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND bearish 12h trend AND volume confirmation
            elif curr_close < S1_aligned[i] and close < ema34_12h_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0