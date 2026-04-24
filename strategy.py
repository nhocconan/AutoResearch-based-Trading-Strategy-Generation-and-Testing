#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for trend filter (price above/below EMA34).
- Entry: Long when price breaks above Donchian(20) high AND price > 1w EMA34 AND volume > 1.5x 20-day average volume.
         Short when price breaks below Donchian(20) low AND price < 1w EMA34 AND volume > 1.5x 20-day average volume.
- Exit: Opposite Donchian breakout OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- Weekly EMA filter ensures we only trade in the direction of the higher timeframe trend.
- Volume confirmation reduces false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~60 total over 4 years (~15/year) based on Donchian breakout frequency with trend and volume filters.
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
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Donchian(20) channels on 1d
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: volume > 1.5x 20-day average volume
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):  # 20-period SMA (0-indexed, need 19 for 20 periods)
        vol_ma20[i] = np.mean(volume[i - 19:i + 1])
    volume_confirm = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need sufficient data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1w EMA34
            if position == 1:
                if curr_close < donchian_low[i] or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > donchian_high[i] or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + trend filter + volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high AND bullish 1w trend AND volume confirmation
            if curr_close > donchian_high[i] and curr_close > ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND bearish 1w trend AND volume confirmation
            elif curr_close < donchian_low[i] and curr_close < ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0