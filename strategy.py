#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above Camarilla R3 in low chop regime (CHOP < 38.2) with volume confirmation
# Short when price breaks below Camarilla S3 in low chop regime with volume confirmation
# Uses 1d ADX > 25 to confirm trending market, avoiding false breakouts in ranging conditions
# Volume spike > 2.0x 20-period average ensures institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_Camarilla_R3S3_Breakout_1dADX25_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[1:period+1])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smooth(dx, 14)
    
    # Align daily ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla levels use previous day's range
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate Camarilla levels: R3, S3
    range_1d = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = prev_close_1d - (range_1d * 1.1 / 4)
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index (CHOP) regime filter on 1d
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR period
        atr = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if i < period:
                atr[i] = np.nan
            else:
                atr[i] = np.nansum(tr[i-period+1:i+1]) / period
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close_arr)
        ll = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if i < period:
                hh[i] = np.nan
                ll[i] = np.nan
            else:
                hh[i] = np.nanmax(high_arr[i-period+1:i+1])
                ll[i] = np.nanmin(low_arr[i-period+1:i+1])
        
        # Chop = LOG10(SUM(TR14)/(HHV(MAX)-LLV(MIN))) * 100 / LOG10(period)
        sum_tr = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if i < period:
                sum_tr[i] = np.nan
            else:
                sum_tr[i] = np.nansum(tr[i-period+1:i+1])
        
        denominator = hh - ll
        chop = np.where((denominator != 0) & (~np.isnan(sum_tr)) & (~np.isnan(denominator)),
                        np.log10(sum_tr / denominator) * 100 / np.log10(period), 50)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for ADX and Camarilla
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_adx = adx_aligned[i]
        curr_chop = chop_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Regime filter: only trade in trending markets (ADX > 25) AND low chop (CHOP < 38.2)
        is_trending = curr_adx > 25
        is_low_chop = curr_chop < 38.2
        is_favorable_regime = is_trending and is_low_chop
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in favorable regime
            if is_favorable_regime and curr_volume_confirm:
                # Bullish breakout: price breaks above Camarilla R3
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Camarilla S3
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to Camarilla H3 level (mean reversion)
            # Calculate H3: C + (H-L)*1.1/2
            prev_close_1d_i = np.concatenate([[np.nan], close_1d[:-1]])[searchsorted_safe(df_1d.index, prices.index[i])] if i < len(prices) else np.nan
            # Simpler: exit at midpoint between R3 and S3
            midpoint = (curr_r3 + curr_s3) / 2.0
            
            if curr_close <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to midpoint between R3 and S3
            midpoint = (curr_r3 + curr_s3) / 2.0
            
            if curr_close >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def searchsorted_safe(arr, val):
    """Helper to find index in array for timestamp"""
    try:
        return np.searchsorted(arr, val, side='right') - 1
    except:
        return 0