#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for trend filter (price above/below EMA50).
- Entry: Long when price breaks above Donchian(20) high AND volume > 1.5x volume MA(20) AND price > 1w EMA50.
         Short when price breaks below Donchian(20) low AND volume > 1.5x volume MA(20) AND price < 1w EMA50.
- Exit: Opposite Donchian breakout OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide objective breakout levels with built-in trend following.
- Volume confirmation ensures breakouts have participation.
- 1w EMA50 filter ensures we only trade with the major trend, reducing whipsaws in sideways markets.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~60 total over 4 years (~15/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper=rolling max(high), lower=rolling min(low)."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Donchian(20) channels
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Donchian/volume MA/EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1w EMA50
            if position == 1:
                if curr_close < lower_20[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1w EMA50
            elif position == -1:
                if curr_close > upper_20[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + volume confirmation + 1w trend filter
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmed AND bullish 1w trend
            if curr_close > upper_20[i] and volume_confirmed[i] and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume confirmed AND bearish 1w trend
            elif curr_close < lower_20[i] and volume_confirmed[i] and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_VolumeSpike_1wEMA50_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0