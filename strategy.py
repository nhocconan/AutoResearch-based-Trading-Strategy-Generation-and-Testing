#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX Trend Filter
# Uses 1d ADX(14) to define trend regime: ADX>25 = strong trend (fade reversals), ADX<20 = ranging/weak trend (trade reversals)
# Williams %R(14) on 6h: Long when %R crosses above -80 from oversold, Short when %R crosses below -20 from overbought
# Only trade reversals in weak/range regime (ADX<20) to avoid trend exhaustion false signals
# Designed for low frequency (50-150 trades over 4 years) with clear reversal logic in appropriate regimes

name = "6h_WilliamsR_Extreme_Reversal_1dADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d HTF data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX(14) calculation for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h Williams %R(14)
    def williams_r(high, low, close, period):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    wr_long_signal = (wr > -80) & (np.concatenate([[False], wr[:-1]]) <= -80)
    wr_short_signal = (wr < -20) & (np.concatenate([[False], wr[:-1]]) >= -20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 14)  # Need ADX and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or 
            np.isnan(wr_long_signal[i]) or np.isnan(wr_short_signal[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade reversals in weak trend/ranging regime
        weak_trend = adx_aligned[i] < 20  # Range/weak trend - trade reversals
        strong_trend = adx_aligned[i] > 25  # Strong trend - avoid reversals (trade with trend instead, but we focus on reversals only)
        
        if position == 0:  # Flat - look for new entries
            if weak_trend:
                # Long reversal: Williams %R crosses above -80 from oversold
                if wr_long_signal[i]:
                    signals[i] = 0.25
                    position = 1
                # Short reversal: Williams %R crosses below -20 from overbought
                elif wr_short_signal[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # In strong trend or transition - stay flat to avoid false reversal signals
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: Williams %R crosses below -50 (momentum loss) or strong trend develops
            exit_long = False
            if wr[i] < -50:  # Loss of bullish momentum
                exit_long = True
            elif adx_aligned[i] > 25:  # Strong trend developed - may not be good for reversal
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Williams %R crosses above -50 (momentum loss) or strong trend develops
            exit_short = False
            if wr[i] > -50:  # Loss of bearish momentum
                exit_short = True
            elif adx_aligned[i] > 25:  # Strong trend developed - may not be good for reversal
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals