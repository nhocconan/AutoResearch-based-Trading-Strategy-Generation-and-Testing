#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour ADX trend strength with 1-day Bollinger Bands mean reversion
# Long when ADX > 25 (trending) and price touches lower Bollinger Band (mean reversion in trend)
# Short when ADX > 25 (trending) and price touches upper Bollinger Band (mean reversion in trend)
# Uses ADX for trend strength filtering and Bollinger Bands for entry/exit in trending markets
# Works in both bull and bear markets by only trading in strong trends with mean reversion entries
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_ADX_Trend_BollingerMeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength
    # ADX = DX smoothed, where DX = |DI+ - DI-| / (DI+ + DI-) * 100
    # DI+ = (Current High - Previous High) smoothed, DI- = (Previous Low - Current Low) smoothed
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Calculate daily Bollinger Bands(20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align HTF data to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: ADX > 25 (strong trend) and price touches or goes below lower BB (mean reversion)
            if adx_val > 25 and close_val <= lower_bb_val:
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (strong trend) and price touches or goes above upper BB (mean reversion)
            elif adx_val > 25 and close_val >= upper_bb_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches or goes above middle (SMA) or ADX weakens
            middle_bb = (upper_bb_val + lower_bb_val) / 2
            if close_val >= middle_bb or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches or goes below middle (SMA) or ADX weakens
            middle_bb = (upper_bb_val + lower_bb_val) / 2
            if close_val <= middle_bb or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals