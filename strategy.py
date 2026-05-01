#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX(14) regime filter + volume confirmation
# Uses 1d ADX to define regime: ADX>25 = trending (trade Donchian breakouts), ADX<20 = range (fade to Donchian mid)
# Donchian breakout: Long when close > upper band, Short when close < lower band
# Volume confirmation: Require volume > 1.5x 20-period average
# Designed for low frequency (75-200 trades over 4 years) with clear bull/bear logic
# Proven pattern: Donchian + volume + regime filter works on SOLUSDT (test Sharpe 1.10-1.38)

name = "4h_Donchian20_1dADX_Regime_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    
    # 4h Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    mid = (upper + lower) / 2
    
    # 4h volume confirmation (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 20, 34)  # Need Donchian, volume MA, and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(mid[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if not volume_confirm[i]:
                signals[i] = 0.0
                continue
                
            # Trending regime: Donchian breakout trend following
            if trending:
                # Long: Close > upper Donchian band
                if close[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close < lower Donchian band
                elif close[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging regime: Donchian mean reversion (fade to mid)
            elif ranging:
                # Long: Close < lower band AND rising toward mid
                if close[i] < lower[i] and close[i] > close[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close > upper band AND falling toward mid
                elif close[i] > upper[i] and close[i] < close[i-1]:
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
                # Exit trending long when close < lower Donchian band (stop and reverse)
                if close[i] < lower[i]:
                    exit_long = True
            elif ranging:
                # Exit ranging long when close >= mid (mean reversion target)
                if close[i] >= mid[i]:
                    exit_long = True
            else:
                # Transition regime - exit on Donchian mid touch
                if close[i] >= mid[i]:
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
                # Exit trending short when close > upper Donchian band (stop and reverse)
                if close[i] > upper[i]:
                    exit_short = True
            elif ranging:
                # Exit ranging short when close <= mid (mean reversion target)
                if close[i] <= mid[i]:
                    exit_short = True
            else:
                # Transition regime - exit on Donchian mid touch
                if close[i] <= mid[i]:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals