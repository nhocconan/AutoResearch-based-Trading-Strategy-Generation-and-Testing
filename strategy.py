#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Supertrend with 1d regime filter
# Uses ADX(14) to identify trending markets, Supertrend for direction,
# and 1d ADX regime filter to avoid chop. Works in both bull and bear
# by only taking trades when higher timeframe confirms strong trend.
# Target: 15-35 trades per year (60-140 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.concatenate([[np.nan], high[:-1]])) > 
                          (np.concatenate([[np.nan], low[:-1]]) - low),
                          np.maximum(high - np.concatenate([[np.nan], high[:-1]]), 0), 0)
        dm_minus = np.where((np.concatenate([[np.nan], low[:-1]]) - low) > 
                           (high - np.concatenate([[np.nan], high[:-1]])),
                           np.maximum(np.concatenate([[np.nan], low[:-1]]) - low, 0), 0)
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        # Wilder's smoothing
        atr[period-1] = np.nansum(tr[:period])
        dm_plus_smooth[period-1] = np.nansum(dm_plus[:period])
        dm_minus_smooth[period-1] = np.nansum(dm_minus[:period])
        
        for i in range(period, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        # DI and DX
        di_plus = np.full_like(atr, np.nan)
        di_minus = np.full_like(atr, np.nan)
        dx = np.full_like(atr, np.nan)
        
        di_plus[atr != 0] = (dm_plus_smooth[atr != 0] / atr[atr != 0]) * 100
        di_minus[atr != 0] = (dm_minus_smooth[atr != 0] / atr[atr != 0]) * 100
        dx[(di_plus + di_minus) != 0] = (np.abs(di_plus - di_minus) / (di_plus + di_minus))[ (di_plus + di_minus) != 0] * 100
        
        # ADX
        adx = np.full_like(dx, np.nan)
        adx[2*period-1] = np.nansum(dx[period:2*period]) / period
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    # Calculate 6h Supertrend
    def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
        # ATR
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.full_like(tr, np.nan)
        atr[period-1] = np.nansum(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Basic Upper and Lower Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        # Final Upper and Lower Bands
        final_upper = np.full_like(upper_band, np.nan)
        final_lower = np.full_like(lower_band, np.nan)
        supertrend = np.full_like(close, np.nan)
        
        final_upper[period-1] = upper_band[period-1]
        final_lower[period-1] = lower_band[period-1]
        supertrend[period-1] = hl2[period-1]
        
        for i in range(period, len(close)):
            final_upper[i] = upper_band[i] if (upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
            final_lower[i] = lower_band[i] if (lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
            
            if i == period:
                supertrend[i] = final_upper[i]
            else:
                if supertrend[i-1] == final_upper[i-1]:
                    supertrend[i] = final_lower[i] if close[i] <= final_lower[i] else final_upper[i]
                else:
                    supertrend[i] = final_upper[i] if close[i] >= final_upper[i] else final_lower[i]
        
        return supertrend, atr
    
    # Calculate 1d ADX for regime filter
    def calculate_adx_simple(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        dm_plus = np.where((high - np.concatenate([[np.nan], high[:-1]])) > 
                          (np.concatenate([[np.nan], low[:-1]]) - low),
                          np.maximum(high - np.concatenate([[np.nan], high[:-1]]), 0), 0)
        dm_minus = np.where((np.concatenate([[np.nan], low[:-1]]) - low) > 
                           (high - np.concatenate([[np.nan], high[:-1]])),
                           np.maximum(np.concatenate([[np.nan], low[:-1]]) - low, 0), 0)
        
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        atr[period-1] = np.nansum(tr[:period])
        dm_plus_smooth[period-1] = np.nansum(dm_plus[:period])
        dm_minus_smooth[period-1] = np.nansum(dm_minus[:period])
        
        for i in range(period, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        di_plus = np.full_like(atr, np.nan)
        di_minus = np.full_like(atr, np.nan)
        dx = np.full_like(atr, np.nan)
        
        di_plus[atr != 0] = (dm_plus_smooth[atr != 0] / atr[atr != 0]) * 100
        di_minus[atr != 0] = (dm_minus_smooth[atr != 0] / atr[atr != 0]) * 100
        dx[(di_plus + di_minus) != 0] = (np.abs(di_plus - di_minus) / (di_plus + di_minus))[ (di_plus + di_minus) != 0] * 100
        
        adx = np.full_like(dx, np.nan)
        adx[2*period-1] = np.nansum(dx[period:2*period]) / period
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    # Calculate indicators
    adx_6h = calculate_adx(high, low, close, 14)
    supertrend_6h, atr_6h = calculate_supertrend(high, low, close, 10, 3.0)
    adx_1d = calculate_adx_simple(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate average volume (24-period = 6 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_6h[i]) or np.isnan(supertrend_6h[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_6h[i]
        st_val = supertrend_6h[i]
        adx_1d_val = adx_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # Regime filter: 1d ADX > 20 indicates trending market
        regime_filter = adx_1d_val > 20
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + price above Supertrend + volume confirmation + regime filter
            if (adx_val > 25 and
                price > st_val and
                volume_confirm and
                regime_filter):
                position = 1
                signals[i] = position_size
            # Short: ADX > 25 (strong trend) + price below Supertrend + volume confirmation + regime filter
            elif (adx_val > 25 and
                  price < st_val and
                  volume_confirm and
                  regime_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: ADX weakens (<20) or price crosses below Supertrend
            if (adx_val < 20 or
                price < st_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: ADX weakens (<20) or price crosses above Supertrend
            if (adx_val < 20 or
                price > st_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ADX_Supertrend_Regime"
timeframe = "6h"
leverage = 1.0