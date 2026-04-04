#!/usr/bin/env python3
"""
Experiment #2711: 6h Williams %R + 1d ADX Trend Filter + Volume Confirmation
HYPOTHESIS: Williams %R identifies overbought/oversold extremes on 6h, while 1d ADX > 25 filters for trending markets to avoid whipsaws in ranging conditions. Volume spike (>2x MA20) confirms institutional participation. This combination should work in both bull (trend continuation) and bear (mean reversion in strong trends) markets by only trading in the direction of the 1d trend when momentum is extreme. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2711_6h_willr1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original length
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # Find first valid value
            start_idx = period
            if np.isnan(data[start_idx]):
                # Find first non-nan
                for i in range(start_idx, len(data)):
                    if not np.isnan(data[i]):
                        start_idx = i
                        break
            if start_idx >= len(data):
                return result
            # First value is simple average
            result[start_idx] = np.nanmean(data[start_idx-period+1:start_idx+1])
            # Subsequent values: Wilder smoothing
            for i in range(start_idx + 1, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        atr = WilderSmoothing(tr, period)
        dm_plus_smooth = WilderSmoothing(dm_plus, period)
        dm_minus_smooth = WilderSmoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = WilderSmoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d trend direction using EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Williams %R(14), Volume MA(20) ===
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = np.where((highest_14 - lowest_14) != 0,
                     ((highest_14 - close) / (highest_14 - lowest_14)) * -100,
                     -50)  # neutral when range is zero
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(willr[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: Williams %R returns from extreme OR ADX weakens
            if position_side > 0:  # Long
                # Exit long when Williams %R rises above -20 (overbought) OR ADX < 20 (weak trend)
                if willr[i] > -20 or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit short when Williams %R falls below -80 (oversold) OR ADX < 20 (weak trend)
                if willr[i] < -80 or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d ADX > 25 for trending market filter
        strong_trend = adx_1d_aligned[i] > 25
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if strong_trend and volume_spike:
            trend_dir = trend_1d_aligned[i]
            # Long entry: Williams %R oversold (< -80) in uptrend
            if trend_dir > 0 and willr[i] < -80:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Williams %R overbought (> -20) in downtrend
            elif trend_dir < 0 and willr[i] > -20:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals