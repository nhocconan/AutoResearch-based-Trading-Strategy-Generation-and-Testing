#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d ADX Regime Filter and Volume Spike.
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime and Williams %R.
- Williams %R(14) identifies overbought/oversold: long when crosses above -80 from below,
  short when crosses below -20 from above.
- Regime filter: Only trade mean reversion when 1d ADX(14) < 20 (range market),
  avoid trending markets where mean reversion fails.
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold bounces in range, in bear via selling overbought bounces in range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Williams %R and ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate ADX(14) on 1d for regime filter
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    up_move = pd.Series(high_1d).diff().values
    down_move = -pd.Series(low_1d).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = pd.Series(high_1d).diff().abs().values
    tr2 = pd.Series(low_1d).diff().abs().values
    tr3 = pd.Series(close_1d).diff().abs().values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, 14)
    
    # Regime: ADX < 20 = range (good for mean reversion), ADX >= 20 = trend (avoid)
    range_regime = adx < 20
    
    # Align Williams %R and regime to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    range_regime_aligned = align_htf_to_ltf(prices, df_1d, range_regime.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Williams %R needs 14 + smoothing, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(range_regime_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in range regime (ADX < 20 on 1d)
        in_range = range_regime_aligned[i] > 0.5
        
        if position == 0 and in_range:
            # Williams %R signals: long when crosses above -80 from below,
            # short when crosses below -20 from above
            if i > 0 and not np.isnan(williams_r_aligned[i-1]):
                wr_prev = williams_r_aligned[i-1]
                wr_curr = williams_r_aligned[i]
                
                # Long signal: Williams %R crosses above -80 from below
                if wr_prev <= -80 and wr_curr > -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short signal: Williams %R crosses below -20 from above
                elif wr_prev >= -20 and wr_curr < -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) or opposite signal
            if i > 0 and not np.isnan(williams_r_aligned[i-1]):
                wr_prev = williams_r_aligned[i-1]
                wr_curr = williams_r_aligned[i]
                if wr_prev < -20 and wr_curr >= -20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) or opposite signal
            if i > 0 and not np.isnan(williams_r_aligned[i-1]):
                wr_prev = williams_r_aligned[i-1]
                wr_curr = williams_r_aligned[i]
                if wr_prev > -80 and wr_curr <= -80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0