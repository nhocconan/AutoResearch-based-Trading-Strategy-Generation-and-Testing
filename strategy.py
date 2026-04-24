#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume confirmation (volume > 1.5x 20-period average), 1w for trend filter (price above/below 50-week EMA).
- Entry: Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average volume AND price > 1w EMA50.
         Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average volume AND price < 1w EMA50.
- Exit: Opposite Donchian breakout (price crosses Donchian(20) mid-line) OR trend filter violation.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Weekly EMA filter keeps us on the right side of the longer-term trend.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
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
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period average volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=0)
    volume_confirmed = volume > (1.5 * vol_ma_20_aligned)
    
    # Calculate 1w trend filter: price > 50-week EMA for long, < for short
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=0)
    trend_bullish = close > ema50_1w_aligned
    trend_bearish = close < ema50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Donchian/EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirmed[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout (cross mid-line) OR trend filter violation
        if position != 0:
            # Exit long: price falls below Donchian mid-line OR trend turns bearish
            if position == 1:
                if curr_close < donchian_mid[i] or not trend_bullish[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above Donchian mid-line OR trend turns bullish
            elif position == -1:
                if curr_close > donchian_mid[i] or not trend_bearish[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and trend alignment
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmed AND bullish trend
            if curr_close > donchian_high[i] and volume_confirmed[i] and trend_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume confirmed AND bearish trend
            elif curr_close < donchian_low[i] and volume_confirmed[i] and trend_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dVolumeConfirm_1wEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0