#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal + 1d ADX Regime Filter
# Williams %R identifies overbought/oversold conditions. In trending regimes (ADX>25),
# we fade extremes for mean reversion. In ranging regimes (ADX<20), we follow momentum
# as price breaks out of consolidation. Uses discrete sizing (0.25) to limit fee drag.
# Target: 50-150 trades over 4 years (12-37/year) with strong regime adaptation.

name = "6h_WilliamsR_1dADX_Regime_ExtremeReversal_v1"
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
    
    # 6h Williams %R (14-period)
    def williams_r(high, low, close, period):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0,
                      -100 * (highest_high - close) / (highest_high - lowest_low),
                      -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # 6h EMA21 for dynamic exit
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 21)  # Need ADX and EMA21
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Trending regime: Williams %R mean reversion (fade extremes)
            if trending:
                # Long: Williams %R oversold (< -80) and turning up
                if wr[i] < -80 and wr[i] > wr[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought (> -20) and turning down
                elif wr[i] > -20 and wr[i] < wr[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging regime: Williams %R momentum (breakout continuation)
            elif ranging:
                # Long: Williams %R rising from oversold territory
                if wr[i] < -50 and wr[i] > wr[i-1] and wr[i-1] < -80:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R falling from overbought territory
                elif wr[i] > -50 and wr[i] < wr[i-1] and wr[i-1] > -20:
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
                # Exit trending long when Williams %R reaches overbought
                if wr[i] >= -20:
                    exit_long = True
            elif ranging:
                # Exit ranging long when price reaches EMA21 (mean reversion target)
                if close[i] >= ema21[i]:
                    exit_long = True
            else:
                # Transition regime - exit on Williams %R deterioration
                if wr[i] >= -50:
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
                # Exit trending short when Williams %R reaches oversold
                if wr[i] <= -80:
                    exit_short = True
            elif ranging:
                # Exit ranging short when price reaches EMA21 (mean reversion target)
                if close[i] <= ema21[i]:
                    exit_short = True
            else:
                # Transition regime - exit on Williams %R deterioration
                if wr[i] <= -50:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals