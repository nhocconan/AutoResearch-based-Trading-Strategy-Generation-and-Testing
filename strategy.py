#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume spike confirmation.
- Entry: Long when price breaks above Camarilla R3 AND 1d EMA34 slope up AND 1d volume > 1.5 * 20-period average volume.
         Short when price breaks below Camarilla S3 AND 1d EMA34 slope down AND 1d volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla breakout (R4/S4) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels identify intraday support/resistance with institutional relevance.
- R3/S3 breaks often indicate strong momentum when confirmed by volume and trend.
- Volume spike confirms institutional participation.
- 1d EMA34 provides medium-term trend filter to avoid counter-trend trades.
- Works in bull markets (buy R3 breaks in uptrend) and bear markets (sell S3 breaks in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def typical_price(high, low, close):
    """Calculate Typical Price."""
    return (high + low + close) / 3.0

def camarilla_levels(high, low, close):
    """
    Calculate Camarilla Pivot Levels.
    Based on previous day's high, low, close.
    Returns: (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    """
    typical = typical_price(high, low, close)
    range_ = high - low
    
    R4 = typical + range_ * 1.1 / 2
    R3 = typical + range_ * 1.1 / 4
    R2 = typical + range_ * 1.1 / 6
    R1 = typical + range_ * 1.1 / 12
    PP = typical
    S1 = typical - range_ * 1.1 / 12
    S2 = typical - range_ * 1.1 / 6
    S3 = typical - range_ * 1.1 / 4
    S4 = typical - range_ * 1.1 / 2
    
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d EMA34 slope (trend direction)
    ema34_slope = np.diff(ema34_1d_aligned, prepend=ema34_1d_aligned[0])
    
    # Calculate 1d volume average and spike confirmation
    df_1d_vol = get_htf_data(prices, '1d')
    if len(df_1d_vol) < 30:
        return np.zeros(n)
    
    vol_20 = pd.Series(df_1d_vol['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_current = df_1d_vol['volume'].values
    vol_ratio = vol_current / (vol_20 + 1e-10)  # Avoid division by zero
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d_vol, vol_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d data (using previous day's HLC)
    # We need to shift the 1d data by 1 to avoid look-ahead (use previous day's levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using that day's HLC)
    camarilla_data = []
    for i in range(len(high_1d)):
        R4, R3, R2, R1, PP, S1, S2, S3, S4 = camarilla_levels(high_1d[i], low_1d[i], close_1d[i])
        camarilla_data.append([R4, R3, R2, R1, PP, S1, S2, S3, S4])
    
    camarilla_array = np.array(camarilla_data)
    # Extract R3 and S3 levels (index 1 and 7)
    r3_1d = camarilla_array[:, 1]
    s3_1d = camarilla_array[:, 7]
    r4_1d = camarilla_array[:, 0]
    s4_1d = camarilla_array[:, 8]
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels to avoid look-ahead)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d, additional_delay_bars=1)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d, additional_delay_bars=1)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d, additional_delay_bars=1)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_slope[i]) or
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below S3 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < s3_1d_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R3 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > r3_1d_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND bullish 1d EMA34 slope
            if curr_close > r3_1d_aligned[i] and vol_ratio_aligned[i] > 1.5 and ema34_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume spike AND bearish 1d EMA34 slope
            elif curr_close < s3_1d_aligned[i] and vol_ratio_aligned[i] > 1.5 and ema34_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0