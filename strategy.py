#!/usr/bin/env python3
# 6h_12h_1d_market_regime_v1
# Hypothesis: Combine 12h market regime (ADX trend strength) with 1d mean reversion (RSI extremes) and 6s momentum.
# In trending regime (ADX>25): follow 6s EMA crossover with 12h trend filter.
# In ranging regime (ADX<20): mean revert at RSI extremes with 1d trend filter.
# Uses volume confirmation to filter false signals. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_market_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h ADX for regime detection (trending vs ranging)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value NaN
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(arr[1:period])  # Skip first NaN
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        atr = smooth_wilder(tr, period)
        plus_dm_smooth = smooth_wilder(plus_dm, period)
        minus_dm_smooth = smooth_wilder(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx[:] = np.nan
        di_sum = plus_di + minus_di
        mask = di_sum > 0
        dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
        
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 12h EMA20 for trend direction
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # 1d RSI(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[:] = np.nan
        avg_loss[:] = np.nan
        
        # First average
        if len(close) > period:
            avg_gain[period] = np.nanmean(gain[1:period+1])
            avg_loss[period] = np.nanmean(loss[1:period+1])
            
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d EMA50 for trend filter in ranging regime
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6s EMA crossover for momentum
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(ema20_12h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.3 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Regime classification
        adx_val = adx_12h_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if is_trending:
                # Exit trend following: EMA cross down or ADX weakening
                if ema9[i] < ema21[i] or adx_val < 20:
                    exit_signal = True
            else:  # ranging
                # Exit mean reversion: RSI > 50 or price > EMA50
                if rsi_1d_aligned[i] > 50 or close[i] > ema50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if is_trending:
                # Exit trend following: EMA cross up or ADX weakening
                if ema9[i] > ema21[i] or adx_val < 20:
                    exit_signal = True
            else:  # ranging
                # Exit mean reversion: RSI < 50 or price < EMA50
                if rsi_1d_aligned[i] < 50 or close[i] < ema50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if is_trending and vol_surge:
                # Trend following: EMA crossover with 12h trend filter
                if ema9[i] > ema21[i] and close[i] > ema20_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif ema9[i] < ema21[i] and close[i] < ema20_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif is_ranging and vol_surge:
                # Mean reversion: RSI extremes with 1d trend filter
                if rsi_1d_aligned[i] < 30 and close[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif rsi_1d_aligned[i] > 70 and close[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals