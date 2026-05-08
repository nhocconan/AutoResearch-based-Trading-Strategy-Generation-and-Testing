#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Volume_Weighted_Trend_With_1d_Regime_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_adx = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr_adx
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr_adx
    dx = np.full_like(atr_adx, np.nan)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    adx = wilders_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h volume-weighted price momentum
    # VWAP-like momentum: (close - vwap) / vwap scaled
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den > 0, vwap_num / vwap_den, np.nan)
    
    # Normalized deviation from VWAP
    vwap_dev = np.where(vwap > 0, (close - vwap) / vwap, 0)
    
    # Smooth the deviation to get momentum signal
    vwap_dev_smooth = pd.Series(vwap_dev).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vwap_dev_smooth[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_aligned[i]
        adx_val = adx_aligned[i]
        mom = vwap_dev_smooth[i]
        
        if position == 0:
            # Enter long: price above EMA34, strong trend (ADX > 25), positive momentum
            if close[i] > ema_val and adx_val > 25 and mom > 0.001:
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA34, strong trend (ADX > 25), negative momentum
            elif close[i] < ema_val and adx_val > 25 and mom < -0.001:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below EMA34 OR trend weakens (ADX < 20) OR momentum turns negative
            if (close[i] <= ema_val or adx_val < 20 or mom < -0.0005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above EMA34 OR trend weakens (ADX < 20) OR momentum turns positive
            if (close[i] >= ema_val or adx_val < 20 or mom > 0.0005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines 4h volume-weighted price momentum with 1d trend filters.
# - Uses 4h VWAP deviation momentum for entry timing
# - Uses 1d EMA(34) for trend direction filter
# - Uses 1d ADX(14) for trend strength filter (ADX>25 to enter, ADX<20 to exit)
# - Only trades in strong trending conditions to avoid whipsaw in ranging markets
# - Works in both bull and bear markets by following the established trend
# - Volume weighting reduces false signals from low-volume moves
# - Target: 80-150 total trades over 4 years (20-38/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return