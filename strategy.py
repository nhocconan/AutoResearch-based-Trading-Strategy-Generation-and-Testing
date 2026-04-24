#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and 1d volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for ADX trend strength filter (ADX > 25), 1d for volume confirmation (volume > 2.0 * 20-period average).
- Donchian channels: Upper = 20-period high, Lower = 20-period low.
- Entry: Long when price breaks above Donchian Upper AND 1w ADX > 25 AND 1d volume > 2.0 * 20-period average volume.
         Short when price breaks below Donchian Lower AND 1w ADX > 25 AND 1d volume > 2.0 * 20-period average volume.
- Exit: Opposite Donchian breakout (price crosses opposite channel) OR ADX falls below 20 (trend weakening).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian breakouts capture strong momentum moves; ADX filter ensures we only trade in trending markets.
- Volume confirmation validates breakout authenticity, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def adx(high, low, close, period=14):
    """Calculate Average Directional Index with proper min_periods."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        plus_dm[i] = plus_dm[i] if plus_dm[i] > minus_dm[i] else 0
        minus_dm[i] = minus_dm[i] if minus_dm[i] > plus_dm[i] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[period] = np.nansum(tr[1:period+1]) if period < len(tr) else 0
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w ADX for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    adx_1w = adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Donchian(20) + ADX warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_ratio = volume[i] / (pd.Series(volume[max(0, i-19):i+1]).mean() + 1e-10)  # 12h volume ratio
        
        # Exit conditions: opposite Donchian breakout OR ADX falls below 20 (trend weakening)
        if position != 0:
            # Exit long: price breaks below Donchian Lower OR ADX < 20
            if position == 1:
                if curr_close < donchian_lower[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian Upper OR ADX < 20
            elif position == -1:
                if curr_close > donchian_upper[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian Upper AND ADX > 25 AND volume confirmation
            if (curr_close > donchian_upper[i] and 
                adx_1w_aligned[i] > 25 and 
                vol_ratio_1d_aligned[i] > 2.0):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian Lower AND ADX > 25 AND volume confirmation
            elif (curr_close < donchian_lower[i] and 
                  adx_1w_aligned[i] > 25 and 
                  vol_ratio_1d_aligned[i] > 2.0):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_1wADX_TrendFilter_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0