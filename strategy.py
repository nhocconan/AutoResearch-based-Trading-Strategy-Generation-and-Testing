#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h ADX trend filter and volume confirmation
# Long when: price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND 12h ADX > 25 AND volume > 1.5x 20-period MA
# Short when: price breaks below lower BB(20,2) AND BB width < 20th percentile (squeeze) AND 12h ADX > 25 AND volume > 1.5x 20-period MA
# Exit when: price returns to middle BB OR BB width > 50th percentile (squeeze ends) OR ADX < 20
# Bollinger squeeze captures low volatility breakouts, ADX filters for trending regimes, volume confirms conviction
# Timeframe: 6h, HTF: 12h for ADX. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBSqueeze_12hADX_VolumeConfirm"
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
    
    # Bollinger Bands (20,2) on 6h
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = ma_20 + (2 * std_20)
        lower_bb = ma_20 - (2 * std_20)
        middle_bb = ma_20
        bb_width = (upper_bb - lower_bb) / ma_20  # normalized width
    else:
        ma_20 = np.full(n, np.nan)
        std_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        middle_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Bollinger squeeze: width < 20th percentile (low volatility)
    if len(bb_width) >= 50:  # need sufficient data for percentile
        bb_width_series = pd.Series(bb_width)
        bb_width_pct_20 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
        bb_width_pct_50 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.50).values
        squeeze_condition = bb_width < bb_width_pct_20
        squeeze_ends = bb_width > bb_width_pct_50
    else:
        squeeze_condition = np.zeros(n, dtype=bool)
        squeeze_ends = np.zeros(n, dtype=bool)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    if len(high_12h) >= 14:
        # True Range
        tr1 = np.abs(high_12h[1:] - low_12h[1:])
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
        minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, 14)
    else:
        adx = np.full(len(df_12h), np.nan)
    
    # ADX trend filter: ADX > 25 = strong trend
    adx_trend = adx > 25
    adx_weak = adx < 20  # for exit condition
    
    # Align 12h ADX to 6h timeframe
    adx_trend_aligned = align_htf_to_ltf(prices, df_12h, adx_trend.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_12h, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(close[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(middle_bb[i]) or np.isnan(squeeze_condition[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_trend_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper BB + squeeze + strong trend + volume filter
            if (close[i] > upper_bb[i] and 
                squeeze_condition[i] and 
                adx_trend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower BB + squeeze + strong trend + volume filter
            elif (close[i] < lower_bb[i] and 
                  squeeze_condition[i] and 
                  adx_trend_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle BB OR squeeze ends OR trend weakens
            if (close[i] < middle_bb[i] or squeeze_ends[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle BB OR squeeze ends OR trend weakens
            if (close[i] > middle_bb[i] or squeeze_ends[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals