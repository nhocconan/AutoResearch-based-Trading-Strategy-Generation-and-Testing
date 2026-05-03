#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD with 1w EMA50 trend filter and 1d ADX regime filter.
# Uses MACD histogram smoothed by volume (VW-MACD) to reduce false signals in low-volume periods.
# Long when VW-MACD histogram crosses above zero AND 1w EMA50 upward AND 1d ADX > 20 (not ranging).
# Short when VW-MACD histogram crosses below zero AND 1w EMA50 downward AND 1d ADX > 20.
# Exit when histogram crosses back toward zero OR ADX < 15 (strong ranging).
# Designed for 6h timeframe to achieve 50-150 trades over 4 years with volume filtering to avoid chop.
# Volume weighting makes MACD more reliable by emphasizing institutional participation.

name = "6h_VWMACD_1wEMA50_1dADX_Regime"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Wilder's smoothing (EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 1w data for EMA50 (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h VW-MACD (Volume Weighted MACD)
    # Typical price for volume weighting
    typical_price = (high + low + close) / 3
    # Volume-weighted price
    vw_price = np.sum(typical_price * volume) / np.sum(volume) if np.sum(volume) > 0 else typical_price
    # For true VW-MACD, we need EMA of volume-weighted prices
    # Approximate: EMA of typical price weighted by volume ratio
    volume_ratio = volume / (pd.Series(volume).rolling(window=20, min_periods=1).mean().values + 1e-10)
    vw_typical = typical_price * (1 + np.log(volume_ratio))  # Log volume weighting
    
    # Standard MACD on volume-weighted typical price
    ema_fast = pd.Series(vw_typical).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(vw_typical).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(macd_hist[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Trend direction from 1w EMA50
        if i >= 1:
            ema50_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            ema50_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            ema50_up = ema50_down = False
            
        # ADX conditions
        adx_trending = adx_1d_aligned[i] > 20
        adx_ranging = adx_1d_aligned[i] < 15
        
        # MACD histogram crossover
        if i >= 1:
            hist_cross_up = macd_hist[i-1] <= 0 and macd_hist[i] > 0
            hist_cross_down = macd_hist[i-1] >= 0 and macd_hist[i] < 0
            hist_reverse_up = macd_hist[i-1] < 0 and macd_hist[i] > macd_hist[i-1]  # Rising from negative
            hist_reverse_down = macd_hist[i-1] > 0 and macd_hist[i] < macd_hist[i-1]  # Falling from positive
        else:
            hist_cross_up = hist_cross_down = hist_reverse_up = hist_reverse_down = False
        
        if position == 0:
            # Long: MACD hist crosses above zero AND 1w EMA50 up AND trending
            if hist_cross_up and ema50_up and adx_trending:
                signals[i] = 0.25
                position = 1
            # Short: MACD hist crosses below zero AND 1w EMA50 down AND trending
            elif hist_cross_down and ema50_down and adx_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: MACD hist falls toward zero OR ADX becomes ranging
            if macd_hist[i] < 0 or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: MACD hist rises toward zero OR ADX becomes ranging
            if macd_hist[i] > 0 or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals