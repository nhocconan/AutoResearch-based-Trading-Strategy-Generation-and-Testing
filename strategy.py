#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA34 trend filter to capture major trend direction.
- Camarilla levels: Calculated from prior 1d OHLC, R1/S1 are key intraday support/resistance.
- Entry: Long when close crosses above R1 AND price > 1w EMA34 AND volume > 2.0 * 20-period average volume.
         Short when close crosses below S1 AND price < 1w EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla level cross OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide structured price channels that work in both trending and ranging markets.
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC."""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r1 = close + range_val * 1.1 / 12.0
    s1 = close - range_val * 1.1 / 12.0
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 210:  # Need sufficient data for 1w EMA34
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
    
    # Calculate Camarilla levels from prior 1d OHLC
    r1_levels = np.full(n, np.nan)
    s1_levels = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use prior bar's OHLC to calculate today's Camarilla levels
        r1, s1 = calculate_camarilla(high[i-1], low[i-1], close[i-1])
        r1_levels[i] = r1
        s1_levels[i] = s1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 1w EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]) or
            np.isnan(r1_levels[i]) or np.isnan(s1_levels[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_r1 = r1_levels[i]
        curr_s1 = s1_levels[i]
        
        # Exit conditions: opposite Camarilla level cross OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price falls below S1 OR price falls below 1w EMA34
            if position == 1:
                if curr_close < curr_s1 or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above R1 OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > curr_r1 or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Long: Close crosses above R1 AND price > 1w EMA34 AND volume confirmation
            long_condition = (curr_close > curr_r1 and 
                            close[i-1] <= r1_levels[i-1] and  # Prior close was at or below R1
                            curr_close > ema34_1w_aligned[i] and
                            curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            # Short: Close crosses below S1 AND price < 1w EMA34 AND volume confirmation
            short_condition = (curr_close < curr_s1 and
                             close[i-1] >= s1_levels[i-1] and  # Prior close was at or above S1
                             curr_close < ema34_1w_aligned[i] and
                             curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
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