#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Camarilla R1 level + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S1 level + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla pivots provide intraday support/resistance levels that work well on 12h timeframe.
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Volume confirmation (1.5x) targets ~25-35 trades/year on 12h timeframe to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) / (period-1)
        # Subsequent values: smoothed
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smma(tr, 14)
    dm_plus_smooth = smma(dm_plus, 14)
    dm_minus_smooth = smma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smma(dx, 14)
    
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h Camarilla Pivot Levels (based on previous day) ===
    # Calculate pivot points using previous 12h bar's OHLC
    # Camarilla formulas:
    # Pivot = (high + low + close) / 3
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    # We use the previous bar's values to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # ADX(14) needs ~50 bars for stability + Donchian-like + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trend_filter = adx_14_1d_aligned[i] > 25.0
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 level
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1[i]) and \
           trend_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 level
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1[i]) and \
             trend_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dADX_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0