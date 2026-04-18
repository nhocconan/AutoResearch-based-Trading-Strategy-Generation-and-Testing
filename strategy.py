#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian breakout with weekly ADX trend filter and volume confirmation.
# Uses weekly Donchian channels (20-period) for breakout signals, weekly ADX (>25) to ensure
# trending markets, and volume confirmation for conviction. Designed for low trade frequency
# (7-25/year) to minimize fee drag. Works in bull markets (breakouts above upper band) and
# bear markets (breakouts below lower band) by only trading in the direction of the weekly trend.
name = "1d_WeeklyDonchian20_ADX_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for indicators (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period) using previous period's data
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    vol_w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period) - using prior week's data to avoid look-ahead
    high_20_w = pd.Series(high_w).rolling(window=20, min_periods=20).max().shift(1).values
    low_20_w = pd.Series(low_w).rolling(window=20, min_periods=20).min().shift(1).values
    upper_band_w = high_20_w
    lower_band_w = low_20_w
    
    # Weekly ADX (14-period) for trend strength
    # True Range calculation
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    minus_dm = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_w = wilders_smoothing(tr_w, 14)
    plus_di_w = wilders_smoothing(plus_dm, 14) / atr_w * 100
    minus_di_w = wilders_smoothing(minus_dm, 14) / atr_w * 100
    dx_w = np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w) * 100
    adx_w = wilders_smoothing(dx_w, 14)
    
    # Weekly average volume for confirmation
    vol_ma_w = pd.Series(vol_w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    upper_band_w_aligned = align_htf_to_ltf(prices, df_1w, upper_band_w)
    lower_band_w_aligned = align_htf_to_ltf(prices, df_1w, lower_band_w)
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    vol_ma_w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_w)
    
    # Session filter: 08-20 UTC (applied to daily data)
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_w_aligned[i]) or np.isnan(lower_band_w_aligned[i]) or
            np.isnan(adx_w_aligned[i]) or np.isnan(vol_ma_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above weekly average
        vol_confirm = volume[i] > vol_ma_w_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_w_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper band AND volume confirmation AND trend filter
            long_breakout = close[i] > upper_band_w_aligned[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < lower_band_w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < lower_band_w_aligned[i] or adx_w_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > upper_band_w_aligned[i] or adx_w_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals