#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout + 1d ADX Trend + Volume Confirmation
# Hypothesis: Donchian channel breakouts with ADX trend strength filter and volume confirmation
# capture institutional breakouts while avoiding false signals in ranging markets.
# Works in bull via upward breakouts, in bear via downward breakdowns, and avoids whipsaws
# in ranges via ADX < 20 filter. Target: 20-40 trades/year for low friction.
name = "6h_donchian20_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - prev_close)
    tr3 = abs(df_1d['low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM = max(high - prev_high, 0) if > max(prev_low - low, 0) else 0
    # -DM = max(prev_low - low, 0) if > max(high - prev_high, 0) else 0
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    up_move = df_1d['high'] - prev_high
    down_move = prev_low - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(series, period):
        result = np.full_like(series, np.nan)
        if len(series) < period:
            return result
        # First value is simple average
        result[period-1] = series.iloc[period-1:period*2-1].mean() if len(series) >= 2*period-1 else series[:period].mean()
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(series)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + series.iloc[i] * alpha
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(pd.Series(plus_dm, index=df_1d.index), 14) / atr
    minus_di = 100 * wilder_smooth(pd.Series(minus_dm, index=df_1d.index), 14) / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(pd.Series(dx, index=df_1d.index), 14)
    
    # Align ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Donchian channel (20-period) on 6h data
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(adx_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or ADX weakens (< 20)
            if close[i] < lower[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or ADX weakens (< 20)
            if close[i] > upper[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian upper with ADX > 25 and volume
            if close[i] > upper[i] and adx_6h[i] > 25 and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower with ADX > 25 and volume
            elif close[i] < lower[i] and adx_6h[i] > 25 and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals