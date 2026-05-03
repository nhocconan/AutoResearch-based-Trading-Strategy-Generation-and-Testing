#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes breakouts; breakout direction confirmed by 1d ADX > 25
# and volume spike. Works in both bull and bear markets by trading volatility expansions
# in the direction of the higher timeframe trend. Designed for low trade frequency (12-37/year) on 6h timeframe.

name = "6h_BollingerSqueeze_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr_14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[13] = np.mean(dx[14:28]) if len(dx) >= 28 else np.mean(dx[14:]) if len(dx) > 14 else 0
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Bollinger Bands (20, 2) on 6h data
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2.0 * dev)
    lower_band = basis - (2.0 * dev)
    
    # Bollinger Band Width (normalized)
    bb_width = (upper_band - lower_band) / (basis + 1e-10)
    
    # Bollinger Band Squeeze: BB Width below 20-period rolling mean of BB Width
    bb_width_s = pd.Series(bb_width)
    bb_width_mean = bb_width_s.rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_mean
    
    # Breakout conditions
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Volume confirmation: volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(basis[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if we have a valid squeeze breakout with trend and volume
        is_squeeze_breakout_up = squeeze_condition[i-1] and breakout_up[i] and adx_aligned[i] > 25 and volume_spike[i]
        is_squeeze_breakout_down = squeeze_condition[i-1] and breakout_down[i] and adx_aligned[i] > 25 and volume_spike[i]
        
        if position == 0:
            # Long: Bollinger Band squeeze breakout to upside with ADX > 25 and volume spike
            if is_squeeze_breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: Bollinger Band squeeze breakout to downside with ADX > 25 and volume spike
            elif is_squeeze_breakout_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to basis (mean reversion) or volatility expands excessively
            if close[i] <= basis[i] or bb_width[i] > 2.0 * bb_width_mean[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to basis (mean reversion) or volatility expands excessively
            if close[i] >= basis[i] or bb_width[i] > 2.0 * bb_width_mean[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals