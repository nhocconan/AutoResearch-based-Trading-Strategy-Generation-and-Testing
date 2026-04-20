#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1d Donchian(20) breakout + volume confirmation + ADX(14) trend filter.
# Breakouts above upper Donchian with volume and strong trend (ADX>25) go long.
# Breakouts below lower Donchian with volume and strong trend go short.
# In weak trends (ADX<20), fade the breakouts for mean reversion.
# Uses 1w EMA(40) to filter trades in direction of weekly trend only.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period)
    upper_dc = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, lower_dc)
    
    # Load 1w data for trend filter (EMA40)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Load 1d data for ADX calculation (using daily data for stability)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / np.where(tr14 == 0, 1, tr14)
    di_minus = 100 * dm_minus14 / np.where(tr14 == 0, 1, tr14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_dc_val = upper_dc_aligned[i]
        lower_dc_val = lower_dc_aligned[i]
        ema_trend = ema_40_1w_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_12h = vol_ratio[i]
        
        # Determine market regime from ADX
        strong_trend = adx_val > 25
        weak_trend = adx_val < 20
        
        # Weekly trend filter
        above_weekly_ema = price > ema_trend
        below_weekly_ema = price < ema_trend
        
        # Volume filter: require above-average volume
        vol_filter = vol_ratio_12h > 1.5
        
        if position == 0:
            # Strong trend: breakout continuation
            if strong_trend and vol_filter:
                if price > upper_dc_val and above_weekly_ema:
                    signals[i] = 0.30
                    position = 1
                elif price < lower_dc_val and below_weekly_ema:
                    signals[i] = -0.30
                    position = -1
            # Weak trend: mean reversion at Donchian levels
            elif weak_trend and vol_filter:
                if price < lower_dc_val and price > ema_trend:  # Oversold in uptrend
                    signals[i] = 0.30
                    position = 1
                elif price > upper_dc_val and price < ema_trend:  # Overbought in downtrend
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches opposite Donchian or trend weakens
            if price < lower_dc_val or adx_val < 20 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price reaches opposite Donchian or trend weakens
            if price > upper_dc_val or adx_val < 20 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_1d_Donchian20_Breakout_ADXTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0