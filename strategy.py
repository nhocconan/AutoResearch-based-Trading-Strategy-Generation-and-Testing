#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla Pivot Breakout with 1w EMA34 Trend Filter and Volume Spike.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA34 trend filter to capture major trend direction.
- Camarilla pivot levels: Calculated from previous 1d OHLC (R1, S1, R3, S3) for breakout entries.
- Entry: Long when close crosses above R1 AND price > 1w EMA34 AND volume > 2.0 * 20-period average volume.
         Short when close crosses below S1 AND price < 1w EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla level cross (S1 for longs, R1 for shorts) OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla pivots provide precise intraday support/resistance levels that often act as breakout/breakdown points.
- 1w EMA34 provides strong long-term trend filter to avoid counter-trend trades during major moves.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~60 total over 4 years (~15/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 210:  # Need sufficient data for 1w EMA34 and pivot calculation
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Calculate 1w volume average for confirmation
    if len(df_1w) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = df_1w['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 1w EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels from previous 1d bar
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            rang = prev_high - prev_low
            
            # Camarilla levels
            R1 = prev_close + rang * 1.1 / 12
            S1 = prev_close - rang * 1.1 / 12
            R3 = prev_close + rang * 1.1 / 4
            S3 = prev_close - rang * 1.1 / 4
        else:
            # Not enough data for pivot calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema34 = ema34_1w_aligned[i]
        
        # Exit conditions: opposite Camarilla level cross OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price falls below S1 OR price falls below 1w EMA34
            if position == 1:
                if curr_close < S1 or curr_close < curr_ema34:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above R1 OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > R1 or curr_close > curr_ema34:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla level breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume (using 1w data)
            vol_confirmed = curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            # Long: Close crosses above R1 AND price > 1w EMA34 AND volume confirmation
            if curr_close > R1 and curr_close > curr_ema34 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Close crosses below S1 AND price < 1w EMA34 AND volume confirmation
            elif curr_close < S1 and curr_close < curr_ema34 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0