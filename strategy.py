#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout + 1d ADX trend filter + volume spike
# - Uses 12h Donchian channels (20-period) for breakout signals
# - Uses 1d ADX (14-period) to filter for trending markets (ADX > 25)
# - Requires volume spike (2x 20-period average) for confirmation
# - Enters long when price breaks above 12h upper Donchian band in uptrend with volume
# - Enters short when price breaks below 12h lower Donchian band in downtrend with volume
# - Exits when price crosses back below/above 12h middle band or ADX drops below 20
# - Designed to capture strong trends with volume confirmation in both bull and bear markets
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_12hDonchian_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing for TR, +DM, -DM
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h indicators to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_4h = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Align 1d ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or np.isnan(donchian_mid_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for trending market (ADX > 25) and volume spike
            trending = adx_4h[i] > 25
            
            if trending and volume_spike[i]:
                # Long: price breaks above 12h upper Donchian band
                if close[i] > donchian_high_4h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 12h lower Donchian band
                elif close[i] < donchian_low_4h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 12h middle band OR ADX drops below 20 (trend weakening)
            if close[i] < donchian_mid_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h middle band OR ADX drops below 20 (trend weakening)
            if close[i] > donchian_mid_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals