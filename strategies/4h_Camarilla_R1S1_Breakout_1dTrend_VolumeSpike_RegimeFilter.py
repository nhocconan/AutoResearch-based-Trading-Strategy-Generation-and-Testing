#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND volume > 2.0x 20-period average AND choppiness regime favors trend (CHOP < 61.8). Enter short when price breaks below S1 level AND 1d trend is down (close < EMA34) AND volume spike AND CHOP < 61.8. Uses 1d EMA34 for stronger trend filter and adds choppiness regime filter to avoid whipsaws in ranging markets. Designed for moderate trade frequency (19-50/year) with edge in both bull and bear markets via trend alignment and volatility-based entry. Avoids SOL-only bias by requiring BTC/ETH to show similar behavior.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla Pivot Levels (R1, S1) from 1d data
    # Based on previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + ((high-low)*1.1/12), S1 = close - ((high-low)*1.1/12)
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (use previous 1d bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Choppiness regime filter: CHOP < 61.8 favors trending market (avoid ranging)
    # Calculate CHOP on 4h data using 14-period lookback
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # ATR = smoothed TR (using Wilder's smoothing = EMA with alpha=1/period)
        atr = np.full_like(tr, np.nan, dtype=float)
        if len(tr) >= period:
            # First ATR value is simple average
            atr[period] = np.nanmean(tr[1:period+1])
            # Subsequent values: Wilder's smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Highest high and lowest low over period
        hh = np.full_like(close_arr, np.nan, dtype=float)
        ll = np.full_like(close_arr, np.nan, dtype=float)
        for i in range(period-1, len(close_arr)):
            hh[i] = np.max(high_arr[i-period+1:i+1])
            ll[i] = np.min(low_arr[i-period+1:i+1])
        
        # Chop = 100 * log10(sum(ATR) / (HH - LL)) / log10(period)
        chop = np.full_like(close_arr, np.nan, dtype=float)
        for i in range(period, len(close_arr)):
            atr_sum = np.nansum(atr[i-period+1:i+1])
            hh_ll = hh[i] - ll[i]
            if hh_ll > 0 and atr_sum > 0:
                chop[i] = 100 * np.log10(atr_sum / hh_ll) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_regime = chop < 61.8  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (34), volume MA warmup (20), chop warmup (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above R1 + 1d uptrend + volume spike + trending regime
            long_signal = breakout_above_r1 and trend_uptrend and volume_spike[i] and chop_regime[i]
            
            # Short: price below S1 + 1d downtrend + volume spike + trending regime
            short_signal = breakout_below_s1 and trend_downtrend and volume_spike[i] and chop_regime[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend change to downtrend OR chop regime becomes ranging
            if (close[i] < camarilla_s1_aligned[i] or not trend_uptrend or not chop_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend change to uptrend OR chop regime becomes ranging
            if (close[i] > camarilla_r1_aligned[i] or not trend_downtrend or not chop_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0