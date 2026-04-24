#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla pivot breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-80 total trades over 4 years (7-20/year).
- HTF: 1w for EMA trend filter and volume spike detection.
- Entry: Long when price breaks above Camarilla R3 level AND 1w EMA34 is rising AND 1w volume > 1.5x 20-period average.
         Short when price breaks below Camarilla S3 level AND 1w EMA34 is falling AND 1w volume > 1.5x 20-period average.
- Exit: Opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Camarilla levels provide intraday support/resistance, 1w EMA34 filters trend, volume confirmation avoids false breakouts.
- Estimated trades: ~50 total over 4 years (~12/year) based on strict confluence requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels."""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r3 = pivot + (range_ * 1.1 / 2.0)
    r4 = pivot + (range_ * 1.1)
    s3 = pivot - (range_ * 1.1 / 2.0)
    s4 = pivot - (range_ * 1.1)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    ema34_prev_1w_aligned = align_htf_to_ltf(prices, df_1w, np.roll(ema34_1w, 1), additional_delay_bars=1)
    ema34_prev_1w_aligned[0] = ema34_1w_aligned[0]  # Handle first value
    
    # Calculate 1w volume spike filter
    vol_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_20_1w, additional_delay_bars=1)
    vol_current_1w = pd.Series(df_1w['volume'].values).rolling(window=1, min_periods=1).mean().values
    vol_current_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_current_1w, additional_delay_bars=1)
    volume_ratio = vol_current_1w_aligned / (vol_20_1w_aligned + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels for current day
        r3, r4, s3, s4 = camarilla_pivot(high[i], low[i], close[i])
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_prev_1w_aligned[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_ratio = volume_ratio[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below S3 OR price falls below 1w EMA34
            if position == 1:
                if curr_close < s3 or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R3 OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > r3 or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above R3 AND 1w EMA34 is rising AND volume > 1.5x average
            if curr_close > r3 and ema34_1w_aligned[i] > ema34_prev_1w_aligned[i] and curr_volume_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 1w EMA34 is falling AND volume > 1.5x average
            elif curr_close < s3 and ema34_1w_aligned[i] < ema34_prev_1w_aligned[i] and curr_volume_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_CamarillaBreakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0