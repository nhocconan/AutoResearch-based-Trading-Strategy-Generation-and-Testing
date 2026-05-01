#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Regime Filter
# Williams %R(14) identifies overbought/oversold conditions
# 1d ADX(14) defines regime: ADX>25 = trending (fade extremes), ADX<20 = range (mean revert)
# Long: Williams %R < -80 (oversold) in trending OR Williams %R > -20 (overbought) in range
# Short: Williams %R > -20 (overbought) in trending OR Williams %R < -80 (oversold) in range
# Designed for low frequency (75-200 trades over 4 years) with clear mean reversion logic

name = "4h_WilliamsR_1dADX_Regime_MeanRev_v1"
timeframe = "4h"
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
    
    # 4h Williams %R(14)
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 14)  # Need ADX and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Trending regime: Williams %R mean reversion (fade extremes)
            if trending:
                # Long: Oversold condition
                if wr[i] < -80:
                    signals[i] = 0.25
                    position = 1
                # Short: Overbought condition
                elif wr[i] > -20:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging regime: Williams %R trend following (ride momentum)
            elif ranging:
                # Long: Rising from oversold
                if wr[i] > -80 and wr[i] > wr[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Falling from overbought
                elif wr[i] < -20 and wr[i] < wr[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Transition regime (ADX 20-25) - stay flat
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if trending:
                # Exit trending long when Williams %R rises above -50 (neutral)
                if wr[i] > -50:
                    exit_long = True
            elif ranging:
                # Exit ranging long when Williams %R reaches overbought
                if wr[i] >= -20:
                    exit_long = True
            else:
                # Transition regime - exit on any deterioration
                if wr[i] > -50:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if trending:
                # Exit trending short when Williams %R falls below -50 (neutral)
                if wr[i] < -50:
                    exit_short = True
            elif ranging:
                # Exit ranging short when Williams %R reaches oversold
                if wr[i] <= -80:
                    exit_short = True
            else:
                # Transition regime - exit on any deterioration
                if wr[i] < -50:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals