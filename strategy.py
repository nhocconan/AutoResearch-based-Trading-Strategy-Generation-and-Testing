#!/usr/bin/env python3
# 4h_Bollinger_Band_Squeeze_Breakout_Volume
# Hypothesis: In low volatility (Bollinger Band width < 20th percentile), breakout of Bollinger Bands with volume > 1.5x average signals trend continuation.
# Long when price breaks above upper band, short when breaks below lower band.
# Uses 1d ADX > 25 to confirm trending regime on higher timeframe.
# Designed for 20-50 trades/year to avoid fee drag. Works in bull/bear via volatility breakout logic.

name = "4h_Bollinger_Band_Squeeze_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
    
    bb_std_dev = np.full(n, np.nan)
    for i in range(bb_period, n):
        bb_std_dev[i] = np.std(close[i-bb_period:i])
    
    bb_upper = sma + bb_std * bb_std_dev
    bb_lower = sma - bb_std * bb_std_dev
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (lookback 50 periods)
    bb_width_percentile = np.full(n, np.nan)
    for i in range(50, n):
        window = bb_width[i-50:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bb_width_percentile[i] = (np.sum(valid < bb_width[i]) / len(valid)) * 100
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX components
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
        low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    plus_dm_smooth = wilder_smooth(plus_dm, period_adx)
    minus_dm_smooth = wilder_smooth(minus_dm, period_adx)
    
    plus_di = np.full(len(df_1d), np.nan)
    minus_di = np.full(len(df_1d), np.nan)
    dx = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if tr_smooth[i] > 0:
            plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    adx = wilder_smooth(dx, period_adx)
    adx_1d = adx
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: Bollinger Band width in low volatility (< 20th percentile)
        is_squeeze = bb_width_percentile[i] < 20
        
        if position == 0:
            # Only enter in trending regime (ADX > 25)
            if adx_1d_aligned[i] > 25:
                # Long: Breakout above upper band with volume confirmation
                if close[i] > bb_upper[i] and volume[i] > 1.5 * vol_ma[i] and is_squeeze:
                    signals[i] = 0.25
                    position = 1
                # Short: Breakout below lower band with volume confirmation
                elif close[i] < bb_lower[i] and volume[i] > 1.5 * vol_ma[i] and is_squeeze:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below middle band (SMA) or volatility expands
            if close[i] < sma[i] or bb_width_percentile[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above middle band (SMA) or volatility expands
            if close[i] > sma[i] or bb_width_percentile[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals