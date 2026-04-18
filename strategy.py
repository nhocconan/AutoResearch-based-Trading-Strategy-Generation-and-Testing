#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX + 4h volume-weighted price action with weekly trend filter.
# Uses weekly ADX to filter trend strength, daily ADX for entry signals,
# and 4h volume confirmation to avoid false breakouts. Designed for low trade frequency
# (<25/year) to minimize fee drag in 1d timeframe. Works in trending markets (ADX>25)
# and avoids ranging markets (ADX<20). Weekly trend ensures alignment with higher timeframe momentum.
name = "1d_ADX_Trend_Filter_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for ADX calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly ADX (14-period) for trend filter
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range and Directional Movement
    tr_w = np.maximum(high_w[1:] - low_w[1:], 
                      np.maximum(np.abs(high_w[1:] - close_w[:-1]), 
                                 np.abs(low_w[1:] - close_w[:-1])))
    tr_w = np.concatenate([[np.nan], tr_w])
    
    plus_dm_w = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                         np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    plus_dm_w = np.concatenate([[np.nan], plus_dm_w])
    
    minus_dm_w = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                          np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    minus_dm_w = np.concatenate([[np.nan], minus_dm_w])
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_w = wilders_smooth(tr_w, 14)
    plus_di_w = 100 * wilders_smooth(plus_dm_w, 14) / atr_w
    minus_di_w = 100 * wilders_smooth(minus_dm_w, 14) / atr_w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_w = wilders_smooth(dx_w, 14)
    
    # Align weekly ADX to daily timeframe
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    # Calculate daily ADX (14-period) for entry signals
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    tr_d = np.maximum(high_d[1:] - low_d[1:], 
                      np.maximum(np.abs(high_d[1:] - close_d[:-1]), 
                                 np.abs(low_d[1:] - close_d[:-1])))
    tr_d = np.concatenate([[np.nan], tr_d])
    
    plus_dm_d = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), 
                         np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    plus_dm_d = np.concatenate([[np.nan], plus_dm_d])
    
    minus_dm_d = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), 
                          np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    minus_dm_d = np.concatenate([[np.nan], minus_dm_d])
    
    atr_d = wilders_smooth(tr_d, 14)
    plus_di_d = 100 * wilders_smooth(plus_dm_d, 14) / atr_d
    minus_di_d = 100 * wilders_smooth(minus_dm_d, 14) / atr_d
    dx_d = 100 * np.abs(plus_di_d - minus_di_d) / (plus_di_d + minus_di_d)
    adx_d = wilders_smooth(dx_d, 14)
    
    # Calculate 4h volume-weighted average price (VWAP) for confirmation
    # Get 4h data for VWAP calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate typical price and VWAP for 4h
    tp_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    vwap_4h = np.cumsum(tp_4h * df_4h['volume'].values) / np.cumsum(df_4h['volume'].values)
    vwap_4h = np.where(np.cumsum(df_4h['volume'].values) == 0, np.nan, vwap_4h)
    
    # Align 4h VWAP to daily timeframe (using last 4h bar of each day)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Calculate daily VWAP for additional confirmation
    tp_d = (high_d + low_d + close_d) / 3
    vwap_d = np.cumsum(tp_d * volume) / np.cumsum(volume)
    vwap_d = np.where(np.cumsum(volume) == 0, np.nan, vwap_d)
    
    # Volume confirmation: current volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_w_aligned[i]) or np.isnan(adx_d[i]) or 
            np.isnan(vwap_4h_aligned[i]) or np.isnan(vwap_d[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: only trade when weekly ADX > 25 (strong trend)
        trend_filter = adx_w_aligned[i] > 25
        
        # Daily ADX filter: only enter when ADX > 20 (trending) and < 40 (not overextended)
        adx_filter = adx_d[i] > 20 and adx_d[i] < 40
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Price action: close above/below VWAP for directional bias
        price_vwap = close[i] > vwap_d[i]
        price_vwap_inv = close[i] < vwap_d[i]
        
        if position == 0:
            # Long: weekly trend up, daily ADX trending, volume confirmation, price above VWAP
            if trend_filter and adx_filter and vol_confirm and price_vwap:
                signals[i] = 0.25
                position = 1
            # Short: weekly trend up (momentum), daily ADX trending, volume confirmation, price below VWAP
            elif trend_filter and adx_filter and vol_confirm and price_vwap_inv:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend weakens OR ADX drops below 20 OR price crosses below VWAP
            exit_condition = (adx_w_aligned[i] <= 25) or (adx_d[i] < 20) or (close[i] < vwap_d[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend weakens OR ADX drops below 20 OR price crosses above VWAP
            exit_condition = (adx_w_aligned[i] <= 25) or (adx_d[i] < 20) or (close[i] > vwap_d[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals