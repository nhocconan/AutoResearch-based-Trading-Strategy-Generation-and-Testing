#!/usr/bin/env python3
"""
4h_1d1w_PriceAction_Pullback_V1
Hypothesis: In strong weekly trends (ADX > 25), price pulls back to daily EMA21 during 4h consolidation (low ATR ratio). Enter long at EMA21 bounce with volume confirmation in uptrend, short at EMA21 rejection in downtrend. Uses weekly trend filter and daily EMA21 as dynamic support/resistance. Target: 20-30 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(arr, period):
    """Calculate EMA with proper handling of NaN"""
    result = np.full_like(arr, np.nan)
    if len(arr) < period:
        return result
    multiplier = 2 / (period + 1)
    result[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        if not np.isnan(arr[i]):
            result[i] = (arr[i] * multiplier) + (result[i-1] * (1 - multiplier))
        else:
            result[i] = result[i-1]
    return result

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.full_like(tr, np.nan)
    if len(tr) < period:
        return atr
    atr[period-1] = np.nanmean(tr[1:period])
    for i in range(period, len(tr)):
        if not np.isnan(tr[i]) and not np.isnan(atr[i-1]):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        else:
            atr[i] = atr[i-1]
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Data (HTF for EMA21 and volatility) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA21
    ema21_1d = calculate_ema(close_1d, 21)
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Daily ATR(14) for volatility filter
    atr14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # === Weekly Data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            result[period-1] = np.nansum(arr[1:period])
            for i in range(period, len(arr)):
                if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
                else:
                    result[i] = result[i-1]
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 4h bar's volume for confirmation
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_cond = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: only trade when weekly ADX > 25 (strong trend)
        strong_trend = adx_1w_aligned[i] > 25
        
        # Volatility filter: only trade when ATR ratio > 0.8 (enough volatility)
        atr_ratio = atr14_1d_aligned[i] / close[i] if close[i] > 0 else 0
        vol_filter = atr_ratio > 0.008  # 0.8% minimum volatility
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price near EMA21 support in uptrend with volume
            if (close[i] > ema21_1d_aligned[i] * 0.998 and  # within 0.2% above EMA
                close[i] < ema21_1d_aligned[i] * 1.002 and   # within 0.2% below EMA
                close[i] > close[i-1] and                    # current bar closed higher
                strong_trend and vol_cond and vol_filter):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price near EMA21 resistance in downtrend with volume
            elif (close[i] < ema21_1d_aligned[i] * 1.002 and  # within 0.2% above EMA
                  close[i] > ema21_1d_aligned[i] * 0.998 and   # within 0.2% below EMA
                  close[i] < close[i-1] and                    # current bar closed lower
                  strong_trend and vol_cond and vol_filter):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below EMA21 or trend weakens
            if close[i] < ema21_1d_aligned[i] * 0.995 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above EMA21 or trend weakens
            if close[i] > ema21_1d_aligned[i] * 1.005 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d1w_PriceAction_Pullback_V1"
timeframe = "4h"
leverage = 1.0