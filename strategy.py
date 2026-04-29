#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime Filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending market)
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending market)
# Exit when Elder Ray signals weaken (Bull Power < 0 for longs, Bear Power < 0 for shorts)
# Uses 1d ADX to filter for trending regimes only, avoiding whipsaws in ranging markets
# Elder Ray captures trend strength via price relative to EMA, effective in both bull/bear trends
# Discrete position sizing (0.25) minimizes fee churn. Target: 12-37 trades/year on 6h timeframe.

name = "6h_ElderRay_1dADX25_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate EMA13 on 6h close for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate ADX on 1d data
    # ADX requires +DI, -DI, and DX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial TR and DM sums (first period)
    tr_sum = np.nansum(tr[:period]) if len(tr) >= period else 0
    plus_dm_sum = np.nansum(plus_dm[:period]) if len(plus_dm) >= period else 0
    minus_dm_sum = np.nansum(minus_dm[:period]) if len(minus_dm) >= period else 0
    
    # Arrays to store smoothed values
    tr_smooth = np.full_like(tr, np.nan, dtype=float)
    plus_dm_smooth = np.full_like(plus_dm, np.nan, dtype=float)
    minus_dm_smooth = np.full_like(minus_dm, np.nan, dtype=float)
    
    # Set initial values
    if len(tr) >= period:
        tr_smooth[period-1] = tr_sum
        plus_dm_smooth[period-1] = plus_dm_sum
        minus_dm_smooth[period-1] = minus_dm_sum
        
        # Wilder's smoothing for subsequent periods
        for i in range(period, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # Calculate DI and DX
    plus_di = np.full_like(tr_smooth, np.nan, dtype=float)
    minus_di = np.full_like(tr_smooth, np.nan, dtype=float)
    dx = np.full_like(tr_smooth, np.nan, dtype=float)
    
    valid = ~np.isnan(tr_smooth) & (tr_smooth != 0)
    plus_di[valid] = (plus_dm_smooth[valid] / tr_smooth[valid]) * 100
    minus_di[valid] = (minus_dm_smooth[valid] / tr_smooth[valid]) * 100
    
    dx_num = np.abs(plus_di - minus_di)
    dx_den = plus_di + minus_di
    dx_valid = valid & (dx_den != 0)
    dx[dx_valid] = (dx_num[dx_valid] / dx_den[dx_valid]) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.full_like(dx, np.nan, dtype=float)
    dx_sum = np.nansum(dx[period:2*period]) if len(dx) >= 2*period else 0
    if len(dx) >= 2*period:
        adx[2*period-1] = dx_sum / period
        for i in range(2*period, len(dx)):
            if not np.isnan(adx[i-1]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 1d indicators to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 2*period)  # ADX needs 2*period for valid value
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_ema13 = ema13_aligned[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_adx = adx_aligned[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power weakens (< 0) or ADX falls below trend threshold
            if curr_bull < 0 or curr_adx < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power weakens (< 0) or ADX falls below trend threshold
            if curr_bear < 0 or curr_adx < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter in trending markets (ADX > 25)
            if curr_adx > 25:
                # Long when Bull Power > 0 AND Bear Power < 0 (strong bullish momentum)
                if curr_bull > 0 and curr_bear < 0:
                    signals[i] = 0.25
                    position = 1
                # Short when Bear Power > 0 AND Bull Power < 0 (strong bearish momentum)
                elif curr_bear > 0 and curr_bull < 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals