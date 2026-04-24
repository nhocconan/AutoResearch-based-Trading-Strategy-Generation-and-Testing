#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX regime filter (ADX > 25 = trending market).
- Entry: Long when price breaks above Donchian upper(20) AND ADX > 25 AND volume > 1.5 * volume SMA(20).
         Short when price breaks below Donchian lower(20) AND ADX > 25 AND volume > 1.5 * volume SMA(20).
- Exit: Opposite Donchian breakout OR ADX < 20 (regime shift to ranging).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in stoploss via opposite break.
- ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranges.
- Volume confirmation adds conviction to breakouts.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    atr = np.zeros_like(high)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_values = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    upper_12h, lower_12h = donchian_channels(high, low, 20)
    
    # Calculate 1d ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=0)
    
    # Volume confirmation: volume > 1.5 * volume SMA(20)
    vol_sma_20 = sma(volume, 20)
    volume_confirmed = volume > (1.5 * vol_sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Donchian/ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR ADX < 20 (regime shift to ranging)
        if position != 0:
            # Exit long: price breaks below lower Donchian OR ADX < 20
            if position == 1:
                if curr_close < lower_12h[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper Donchian OR ADX < 20
            elif position == -1:
                if curr_close > upper_12h[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + ADX > 25 + volume confirmation
        if position == 0:
            # Long: price breaks above upper Donchian AND ADX > 25 AND volume confirmed
            if curr_close > upper_12h[i] and adx_1d_aligned[i] > 25 and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND ADX > 25 AND volume confirmed
            elif curr_close < lower_12h[i] and adx_1d_aligned[i] > 25 and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_Regime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0