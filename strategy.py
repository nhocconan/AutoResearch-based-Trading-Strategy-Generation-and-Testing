#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 4h Donchian(20) breakout + volume confirmation + ADX(14) trend filter.
# Breakouts above upper Donchian with volume and strong trend (ADX>25) go long.
# Breakouts below lower Donchian with volume and strong trend go short.
# Uses 4h EMA(50) to filter trades in direction of intermediate trend.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for Donchian channels (self-referential for primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels (20-period)
    upper_dc = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Load 4h data for EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 4h data for ADX calculation
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 4h data
    high_4h_arr = df_4h['high'].values
    low_4h_arr = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    # True Range
    tr1 = high_4h_arr - low_4h_arr
    tr2 = np.abs(high_4h_arr - np.roll(close_4h_arr, 1))
    tr3 = np.abs(low_4h_arr - np.roll(close_4h_arr, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_4h_arr - np.roll(high_4h_arr, 1)) > (np.roll(low_4h_arr, 1) - low_4h_arr),
                       np.maximum(high_4h_arr - np.roll(high_4h_arr, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h_arr, 1) - low_4h_arr) > (high_4h_arr - np.roll(high_4h_arr, 1)),
                        np.maximum(np.roll(low_4h_arr, 1) - low_4h_arr, 0), 0)
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
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(adx[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_dc_val = upper_dc[i]
        lower_dc_val = lower_dc[i]
        ema_trend = ema_50_4h[i]
        adx_val = adx[i]
        vol_ratio_4h = vol_ratio[i]
        
        # Determine market regime from ADX
        strong_trend = adx_val > 25
        weak_trend = adx_val < 20
        
        # Intermediate trend filter
        above_ema = price > ema_trend
        below_ema = price < ema_trend
        
        # Volume filter: require above-average volume
        vol_filter = vol_ratio_4h > 1.5
        
        if position == 0:
            # Strong trend: breakout continuation
            if strong_trend and vol_filter:
                if price > upper_dc_val and above_ema:
                    signals[i] = 0.30
                    position = 1
                elif price < lower_dc_val and below_ema:
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

name = "4h_Donchian20_EMA50_ADX_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0