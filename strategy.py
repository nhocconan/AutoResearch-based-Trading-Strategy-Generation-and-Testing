#!/usr/bin/env python3
# 4h_4H_Supertrend_Trend_Filter_1dVWAP_Mean_Reversion
# Hypothesis: In ranging markets (identified by ADX < 25), price tends to revert to the daily VWAP.
# Supertrend (ATR=10, multiplier=3) on 4h provides trend direction to avoid counter-trend trades in strong trends.
# Combines mean reversion in ranging markets with trend filter to avoid whipsaws.
# Works in bull/bear: VWAP acts as dynamic support/resistance, Supertrend avoids false signals in trends.

name = "4h_4H_Supertrend_Trend_Filter_1dVWAP_Mean_Reversion"
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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Align VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate ADX for regime filtering (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # Initial average
            atr[period-1] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI+ and DI-
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        valid = (~np.isnan(atr)) & (atr != 0)
        di_plus[valid] = (dm_plus_smooth[valid] / atr[valid]) * 100
        di_minus[valid] = (dm_minus_smooth[valid] / atr[valid]) * 100
        
        # DX and ADX
        dx = np.full_like(di_plus, np.nan)
        di_sum = di_plus + di_minus
        valid_dx = (~np.isnan(di_plus)) & (~np.isnan(di_minus)) & (di_sum != 0)
        dx[valid_dx] = (np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]) * 100
        
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate Supertrend (ATR=10, multiplier=3)
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.full_like(tr, np.nan)
        if len(tr) >= atr_period:
            atr[atr_period-1] = np.nanmean(tr[1:atr_period+1])
            for i in range(atr_period, len(tr)):
                atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
        
        # Basic Upper and Lower Bands
        hl_avg = (high + low) / 2
        upper_basic = hl_avg + multiplier * atr
        lower_basic = hl_avg - multiplier * atr
        
        # Final Upper and Lower Bands
        final_upper = np.full_like(upper_basic, np.nan)
        final_lower = np.full_like(lower_basic, np.nan)
        
        for i in range(1, len(close)):
            if np.isnan(upper_basic[i]) or np.isnan(lower_basic[i]):
                final_upper[i] = np.nan
                final_lower[i] = np.nan
            else:
                if (close[i-1] <= final_upper[i-1] or np.isnan(final_upper[i-1])) and \
                   (close[i-1] >= final_lower[i-1] or np.isnan(final_lower[i-1])):
                    final_upper[i] = upper_basic[i]
                    final_lower[i] = lower_basic[i]
                elif close[i-1] > final_upper[i-1]:
                    final_upper[i] = upper_basic[i]
                    final_lower[i] = final_lower[i-1]
                else:  # close[i-1] < final_lower[i-1]
                    final_upper[i] = final_upper[i-1]
                    final_lower[i] = lower_basic[i]
        
        # Supertrend
        supertrend = np.full_like(close, np.nan)
        for i in range(len(close)):
            if np.isnan(final_upper[i]) or np.isnan(final_lower[i]):
                supertrend[i] = np.nan
            elif i == 0:
                supertrend[i] = final_upper[i]
            else:
                if supertrend[i-1] == final_upper[i-1]:
                    if close[i] <= final_upper[i]:
                        supertrend[i] = final_upper[i]
                    else:
                        supertrend[i] = final_lower[i]
                else:  # supertrend[i-1] == final_lower[i-1]
                    if close[i] >= final_lower[i]:
                        supertrend[i] = final_lower[i]
                    else:
                        supertrend[i] = final_upper[i]
        
        return supertrend, atr
    
    supertrend, atr = calculate_supertrend(high, low, close, 10, 3)
    
    # Volume filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 28)  # Ensure volume MA, ADX, and Supertrend are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(supertrend[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price below VWAP AND ranging market (ADX < 25) AND volume spike
            if (close[i] < vwap_1d_aligned[i] and 
                adx[i] < 25 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price above VWAP AND ranging market (ADX < 25) AND volume spike
            elif (close[i] > vwap_1d_aligned[i] and 
                  adx[i] < 25 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above VWAP OR trend emerges (ADX >= 25) OR Supertrend flips
            if (close[i] > vwap_1d_aligned[i] or 
                adx[i] >= 25 or 
                supertrend[i] > close[i]):  # Supertrend flipped to down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below VWAP OR trend emerges (ADX >= 25) OR Supertrend flips
            if (close[i] < vwap_1d_aligned[i] or 
                adx[i] >= 25 or 
                supertrend[i] < close[i]):  # Supertrend flipped to up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals