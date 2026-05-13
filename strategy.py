#!/usr/bin/env python3
name = "4h_ChaikinBreakout_TrendFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1D data ONCE for ADX and Chaikin Money Flow
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ADX(14) on 1D
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        # First values (simple average)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full_like(atr, np.nan)
        di_minus = np.full_like(atr, np.nan)
        dx = np.full_like(atr, np.nan)
        
        valid = ~np.isnan(atr) & (atr != 0)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
        
        # ADX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate Chaikin Money Flow (CMF) on 1D
    def calculate_cmf(high, low, close, volume, period=20):
        # Money Flow Multiplier
        mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
        # Money Flow Volume
        mfv = mfm * volume
        
        # Sum of MFV and Volume over period
        mfv_sum = np.full_like(close, np.nan)
        volume_sum = np.full_like(close, np.nan)
        
        for i in range(len(close)):
            if i >= period - 1:
                mfv_sum[i] = np.sum(mfv[i - period + 1:i + 1])
                volume_sum[i] = np.sum(volume[i - period + 1:i + 1])
        
        # CMF
        cmf = np.full_like(close, np.nan)
        valid = volume_sum != 0
        cmf[valid] = mfv_sum[valid] / volume_sum[valid]
        return cmf
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    cmf_1d = calculate_cmf(high_1d, low_1d, close_1d, volume_1d, 20)
    
    # Align 1D indicators to 4H timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    cmf_1d_aligned = align_htf_to_ltf(prices, df_1d, cmf_1d)
    
    # 4H Donchian Channel (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(len(high)):
            if i >= period - 1:
                upper[i] = np.max(high[i - period + 1:i + 1])
                lower[i] = np.min(low[i - period + 1:i + 1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(cmf_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        is_trending = adx_1d_aligned[i] > 25
        
        # Money flow filter: CMF > 0 for buying pressure, < 0 for selling pressure
        bullish_money_flow = cmf_1d_aligned[i] > 0.05
        bearish_money_flow = cmf_1d_aligned[i] < -0.05
        
        # Price breakout conditions
        price_above_upper = close[i] > donchian_upper[i]
        price_below_lower = close[i] < donchian_lower[i]
        
        if position == 0:
            # LONG: Uptrend + bullish money flow + break above upper band
            if is_trending and bullish_money_flow and price_above_upper:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + bearish money flow + break below lower band
            elif is_trending and bearish_money_flow and price_below_lower:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakens or money flow turns bearish or price breaks below lower band
            if (adx_1d_aligned[i] <= 20) or (cmf_1d_aligned[i] < -0.05) or (close[i] < donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens or money flow turns bullish or price breaks above upper band
            if (adx_1d_aligned[i] <= 20) or (cmf_1d_aligned[i] > 0.05) or (close[i] > donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals