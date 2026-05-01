#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h volume confirmation and 1d ADX trend filter.
# Long when: BB width at 20-period low (squeeze) + price breaks above upper BB + 12h volume > 1.5x 20-period average + 1d ADX > 25.
# Short when: BB width at 20-period low (squeeze) + price breaks below lower BB + 12h volume > 1.5x 20-period average + 1d ADX > 25.
# Uses discrete sizing 0.25. Target: 15-30 trades/year.
# Bollinger squeeze identifies low volatility periods primed for breakout.
# Volume confirmation ensures breakout legitimacy. ADX filter avoids choppy markets.
# Works in bull (breakouts continuation) and bear (breakdown continuation) by trading with the 1d trend.

name = "6h_BollingerSqueeze_Breakout_12hVolume_1dADX_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = basis + 2.0 * dev
    lower_bb = basis - 2.0 * dev
    bb_width = (upper_bb - lower_bb) / basis  # Normalized width
    
    # Calculate BB width percentile lookback 50 periods for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    squeeze_condition = bb_width_percentile <= 0.1  # Width at or below 10th percentile
    
    # Calculate 12h average volume (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        """Wilder's smoothing (similar to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
            else:
                result[i] = np.nan
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 
                  0)
    adx = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for BB width percentile and ADX
    
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
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(vol_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_upper_bb = upper_bb[i]
        curr_lower_bb = lower_bb[i]
        curr_squeeze = squeeze_condition[i]
        curr_vol_12h = vol_12h_aligned[i]
        curr_adx = adx_1d_aligned[i]
        
        # Volume confirmation: current 12h-aligned volume > 1.5x 20-period average
        volume_confirm = curr_volume > (1.5 * curr_vol_12h)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Squeeze + break above upper BB + volume confirm + ADX > 25
            if (curr_squeeze and 
                curr_close > curr_upper_bb and 
                volume_confirm and 
                curr_adx > 25):
                signals[i] = 0.25
                position = 1
            # Short: Squeeze + break below lower BB + volume confirm + ADX > 25
            elif (curr_squeeze and 
                  curr_close < curr_lower_bb and 
                  volume_confirm and 
                  curr_adx > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below basis (middle BB) OR squeeze breaks (volatility expansion)
            if (curr_close < basis[i] or not curr_squeeze):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above basis (middle BB) OR squeeze breaks (volatility expansion)
            if (curr_close > basis[i] or not curr_squeeze):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals