#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX regime filter.
# Long when: BB width at 20-period low (squeeze) AND price breaks above upper band AND 1d ADX > 25 (trending).
# Short when: BB width at 20-period low (squeeze) AND price breaks below lower band AND 1d ADX > 25 (trending).
# Uses discrete sizing 0.25. BB squeeze identifies low volatility precede breakout; ADX ensures trending environment.
# Works in bull (breakouts up) and bear (breakouts down) by capturing volatility expansion in trending markets.
# Target: 15-30 trades/year.

name = "6h_BB_Squeeze_ADX_Trend_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (trend strength)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        """Wilder's smoothing (similar to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
            else:
                result[i] = np.nan
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, tr_period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, tr_period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, tr_period)  # ADX = smoothed DX
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2.0
    
    # Middle band (SMA)
    close_series = pd.Series(close)
    middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    
    # Standard deviation
    std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Upper and lower bands
    upper = middle + (bb_std * std_dev)
    lower = middle - (bb_std * std_dev)
    
    # Bollinger Band Width (normalized)
    bb_width = (upper - lower) / middle
    
    # BB Width percentile lookback (50-period) to identify squeeze (lowest 10%)
    bb_width_series = pd.Series(bb_width)
    bb_width_rank = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for BB and ADX calculations
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(middle[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bb_width_rank[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_middle = middle[i]
        curr_upper = upper[i]
        curr_lower = lower[i]
        curr_bb_width_rank = bb_width_rank[i]
        curr_adx = adx_aligned[i]
        
        # BB Squeeze condition: BB width in lowest 10% of lookback
        is_squeeze = curr_bb_width_rank <= 0.10
        
        # Breakout conditions
        breakout_up = curr_close > curr_upper
        breakout_down = curr_close < curr_lower
        
        # ADX trend condition: trending market (ADX > 25)
        is_trending = curr_adx > 25
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Squeeze + breakout up + trending
            if is_squeeze and breakout_up and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Squeeze + breakout down + trending
            elif is_squeeze and breakout_down and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below middle band (mean reversion) OR ADX weakens (< 20)
            if curr_close < curr_middle or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above middle band OR ADX weakens (< 20)
            if curr_close > curr_middle or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals