# 4h_HMA_TRIX_Volume_Spice_Trend_Filter
# Hypothesis: 4h TRIX momentum with HMA trend filter and volume spike confirmation.
# Long when TRIX > 0 and price > HMA(21) with volume > 1.5x 20-bar MA.
# Short when TRIX < 0 and price < HMA(21) with volume > 1.5x 20-bar MA.
# Uses 1d ADX > 25 as regime filter to avoid whipsaws in ranging markets.
# Designed for low trade frequency (15-35 trades/year) to minimize fee drag.
# Works in bull markets via momentum and in bear markets via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 
                                   np.abs(high_1d[0] - close_1d[0] if len(close_1d) > 0 else 0),
                                   np.abs(low_1d[0] - close_1d[0] if len(close_1d) > 0 else 0)])], tr])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def _wilder_smooth(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = _wilder_smooth(tr, 14)
    dm_plus_smooth = _wilder_smooth(dm_plus, 14)
    dm_minus_smooth = _wilder_smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = _wilder_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 4h data for HMA and TRIX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate HMA(21) on 4h
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma1 = pd.Series(arr).rolling(window=half, min_periods=half).mean().values
        wma2 = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw = 2 * wma1 - wma2
        hma_vals = pd.Series(raw).rolling(window=sqrt, min_periods=sqrt).mean().values
        return hma_vals
    
    close_4h = df_4h['close'].values
    hma_21 = hma(close_4h, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    # Calculate TRIX(12) on 4h
    def ema(arr, period):
        return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    ema1 = ema(close_4h, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix = np.where(ema2[:-1] != 0, (ema3[1:] - ema2[:-1]) / ema2[:-1] * 100, 0)
    trix = np.concatenate([[np.nan], trix])
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or 
            np.isnan(trix_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending market (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # Entry conditions
        long_entry = (trix_aligned[i] > 0) and (close[i] > hma_21_aligned[i]) and trending and (volume[i] > 1.5 * vol_ma_20[i])
        short_entry = (trix_aligned[i] < 0) and (close[i] < hma_21_aligned[i]) and trending and (volume[i] > 1.5 * vol_ma_20[i])
        
        # Exit conditions: opposite signal or loss of trend
        long_exit = (trix_aligned[i] < 0) or (close[i] < hma_21_aligned[i]) or (not trending)
        short_exit = (trix_aligned[i] > 0) or (close[i] > hma_21_aligned[i]) or (not trending)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_HMA_TRIX_Volume_Spice_Trend_Filter"
timeframe = "4h"
leverage = 1.0