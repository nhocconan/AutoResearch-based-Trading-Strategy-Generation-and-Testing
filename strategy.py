#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian_Breakout_Volume
Hypothesis: ADX regime filter (ADX>25 = trending, ADX<20 = ranging) combined with Donchian(20) breakouts and volume confirmation captures strong trends while avoiding choppy markets. Uses 1d HTF for ADX calculation to reduce noise. Long when price breaks above Donchian upper band in uptrend regime with volume >1.5x average. Short when price breaks below Donchian lower band in downtrend regime with volume confirmation. Exits on opposite Donchian band touch or regime shift. Designed for 6h timeframe targeting 80-120 trades over 4 years (20-30/year). Works in bull markets via upside breakouts and bear markets via downside breakdowns. Volume confirmation prevents false breakouts in low-volume environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation (less noisy on higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilder_smooth(data, alpha):
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, alpha)
    plus_dm_smooth = wilder_smooth(plus_dm, alpha)
    minus_dm_smooth = wilder_smooth(minus_dm, alpha)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, alpha)
    
    # Donchian channels on 6h
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: current volume > 1.5 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need ADX (14*2=28 for smoothing), Donchian (20), volume avg (30)
    start_idx = max(30, 20, 28)  # 30 for volume, 20 for Donchian, 28 for ADX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        vol_conf = volume_confirm[i]
        
        # Determine trend direction from DI crossover
        is_uptrend = plus_di_val > minus_di_val
        is_downtrend = minus_di_val > plus_di_val
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging (avoid entries in ranging)
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Only enter in trending regime
            if is_trending:
                if is_uptrend:
                    # Uptrend: long when price breaks above upper Donchian band with volume
                    if (close_val > upper) and vol_conf:
                        signals[i] = size
                        position = 1
                elif is_downtrend:
                    # Downtrend: short when price breaks below lower Donchian band with volume
                    if (close_val < lower) and vol_conf:
                        signals[i] = -size
                        position = -1
            # Do not enter in ranging regime (ADX < 20) to avoid whipsaws
        elif position == 1:
            # Exit long: price touches lower Donchian band or regime shifts to ranging/downtrend
            exit_condition = (close_val < lower) or (not is_trending) or (is_downtrend and adx_val > 25)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches upper Donchian band or regime shifts to ranging/uptrend
            exit_condition = (close_val > upper) or (not is_trending) or (is_uptrend and adx_val > 25)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_Regime_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0