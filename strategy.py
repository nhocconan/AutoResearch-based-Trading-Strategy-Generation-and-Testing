#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter and volume average calculation.
- Camarilla levels: Calculated from previous 1d OHLC to identify key intraday support/resistance.
- Entry: Long when price breaks above R1 with close > R1 AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below S1 with close < S1 AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Price crosses 1d EMA34 in opposite direction OR opposite Camarilla breakout (R1 for short, S1 for long).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide mathematically derived support/resistance that work well in ranging markets.
- 1d EMA34 provides intermediate trend filter to avoid counter-trend trades.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day."""
    # Camarilla levels based on previous day's OHLC
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    
    # Resistance levels
    r1 = close + range_hl * 1.1 / 12
    r2 = close + range_hl * 1.1 / 6
    r3 = close + range_hl * 1.1 / 4
    r4 = close + range_hl * 1.1 / 2
    
    # Support levels
    s1 = close - range_hl * 1.1 / 12
    s2 = close - range_hl * 1.1 / 6
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for initialization
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # We need to shift the 1d data by 1 to get previous day's levels
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    # Calculate Camarilla levels for each day
    camarilla_data = []
    for i in range(len(prev_high)):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            camarilla_data.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(prev_high[i], prev_low[i], prev_close[i])
            camarilla_data.append((r1, r2, r3, r4, s1, s2, s3, s4))
    
    # Extract R1 and S1 levels (most important for breakouts)
    r1_1d = np.array([x[0] for x in camarilla_data])
    s1_1d = np.array([x[4] for x in camarilla_data])
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d, additional_delay_bars=1)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: price crosses 1d EMA34 in opposite direction OR opposite Camarilla breakout
        if position != 0:
            # Exit long: price falls below 1d EMA34 OR breaks below S1
            if position == 1:
                if curr_close < ema34_1d_aligned[i] or curr_low < s1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above 1d EMA34 OR breaks above R1
            elif position == -1:
                if curr_close > ema34_1d_aligned[i] or curr_high > r1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Bullish breakout: price closes above R1
            bullish_breakout = curr_close > r1_1d_aligned[i]
            # Bearish breakout: price closes below S1
            bearish_breakout = curr_close < s1_1d_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            # Use the aligned volume ratio from 1d timeframe
            vol_confirmed = curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            # Long: Bullish breakout AND price > 1d EMA34 AND volume confirmation
            if bullish_breakout and curr_close > ema34_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bearish breakout AND price < 1d EMA34 AND volume confirmation
            elif bearish_breakout and curr_close < ema34_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0