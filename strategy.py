#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1-week ADX trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# Weekly ADX ensures we only trade in strong trends to avoid whipsaws in ranging markets.
# Volume confirmation adds conviction to reversal signals.
# Designed for low trade frequency (12-37/year) to minimize fee drift in 12h timeframe.
# Works in bull markets (oversold bounces in uptrend) and bear markets (overbought reversals in downtrend).
name = "12h_WilliamsR_WeeklyADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for ADX filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate weekly ADX for trend strength
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI values
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND strong uptrend (ADX > 25) AND volume
            oversold = williams_r[i] < -80
            strong_uptrend = adx_aligned[i] > 25
            
            if vol_confirm and oversold and strong_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND strong downtrend (ADX > 25) AND volume
            elif (vol_confirm and 
                  williams_r[i] > -20 and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) OR ADX weakens (< 20)
            return_to_neutral = williams_r[i] > -50
            weak_trend = adx_aligned[i] < 20
            
            if return_to_neutral or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) OR ADX weakens (< 20)
            return_to_neutral = williams_r[i] < -50
            weak_trend = adx_aligned[i] < 20
            
            if return_to_neutral or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals