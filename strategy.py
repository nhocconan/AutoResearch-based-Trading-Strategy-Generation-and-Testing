#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation
# Uses Bollinger Band width percentile to detect low volatility squeezes (breakout precursors),
# 1d ADX > 25 to confirm trending regime, and volume > 1.5x 20-period average for confirmation.
# Works in both bull and bear markets by only taking breakouts in the direction of the 1d trend.
# Target: 20-40 trades/year to minimize fee decay while capturing explosive moves after low volatility periods.
# Focus on BTC/ETH as primary assets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Rest is Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilders_smooth(tr, period_adx)
    dm_plus_smooth = wilders_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilders_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.full(n_1d, np.nan)
    di_minus = np.full(n_1d, np.nan)
    dx = np.full(n_1d, np.nan)
    
    for i in range(len(tr_smooth)):
        if np.isnan(tr_smooth[i]) or tr_smooth[i] == 0:
            di_plus[i] = 0
            di_minus[i] = 0
        else:
            di_plus[i] = 100 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / tr_smooth[i]
    
    for i in range(len(dx)):
        if np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or (di_plus[i] + di_minus[i]) == 0:
            dx[i] = 0
        else:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX = Wilder's smoothed DX
    adx_1d = wilders_smooth(dx, period_adx)
    
    # Bollinger Bands on 4h (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        std_dev[i] = np.std(close[i-bb_period:i])
        upper_band[i] = sma[i] + bb_std * std_dev[i]
        lower_band[i] = sma[i] - bb_std * std_dev[i]
        if sma[i] > 0:
            bb_width[i] = (upper_band[i] - lower_band[i]) / sma[i] * 100
    
    # Bollinger Band width percentile (lookback 50 periods) to identify squeezes
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback, n):
        if np.isnan(bb_width[i-lookback:i]).any():
            bb_width_percentile[i] = np.nan
        else:
            # Percentage of values in lookback that are <= current value
            bb_width_percentile[i] = np.sum(bb_width[i-lookback:i] <= bb_width[i]) / lookback * 100
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align HTF indicators to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(bb_period, vol_period, lookback) + period_adx
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. Bollinger Band squeeze: width in lowest 20% of recent values (low volatility)
        # 2. ADX > 25: trending regime on 1d
        # 3. Volume confirmation: > 1.5x average volume
        # 4. Breakout direction: price breaks above upper band (long) or below lower band (short)
        squeeze_condition = bb_width_percentile[i] < 20
        trend_condition = adx_1d_aligned[i] > 25
        volume_confirmation = vol_ratio > 1.5
        breakout_up = price > upper_band[i]
        breakout_down = price < lower_band[i]
        
        if position == 0:
            # Long: bullish breakout during squeeze with uptrend and volume
            if squeeze_condition and trend_condition and volume_confirmation and breakout_up:
                signals[i] = size
                position = 1
            # Short: bearish breakout during squeeze with uptrend and volume
            elif squeeze_condition and trend_condition and volume_confirmation and breakout_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to middle of Bollinger Bands or volatility expands
            if price < sma[i] or bb_width_percentile[i] > 80:  # exit at mean or high volatility
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to middle of Bollinger Bands or volatility expands
            if price > sma[i] or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_BollingerSqueeze_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0