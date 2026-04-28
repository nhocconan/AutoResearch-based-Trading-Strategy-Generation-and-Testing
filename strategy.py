#!/usr/bin/env python3
# Hypothesis: 1d Bollinger Band squeeze breakout with weekly ADX trend filter and volume confirmation.
# In low volatility regimes (BB width < 20th percentile), price often breaks out with strong momentum.
# Weekly ADX > 25 ensures we only trade in strong trending markets, avoiding whipsaws in ranges.
# Volume confirmation validates breakout authenticity. Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by filtering for strong trends via weekly ADX and only trading breakouts from low volatility.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate weekly ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Wilder smoothing: new_value = prev_value - (prev_value/period) + current_value/period
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di_smoothed = wilders_smoothing(plus_dm, 14)
    minus_di_smoothed = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Bollinger Bands (20, 2) on daily data
    bb_period = 20
    bb_std = 2
    
    # Calculate rolling mean and std
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + (bb_std_dev * bb_std)
    bb_lower = bb_mid - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width (20-period lookback for percentile)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Squeeze condition: BB width below 20th percentile indicates low volatility
        squeeze_condition = bb_width_percentile[i] < 20
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper[i]
        breakout_down = close[i] < bb_lower[i]
        
        # Entry conditions with volume confirmation
        long_entry = strong_trend and squeeze_condition and breakout_up and volume_filter[i]
        short_entry = strong_trend and squeeze_condition and breakout_down and volume_filter[i]
        
        # Exit conditions: return to middle band or trend weakness
        long_exit = (not strong_trend) or (close[i] < bb_mid[i]) or (position == 1 and close[i] < bb_lower[i])
        short_exit = (not strong_trend) or (close[i] > bb_mid[i]) or (position == -1 and close[i] > bb_upper[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_BollingerSqueeze_1wADX_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0