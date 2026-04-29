#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX regime filter and volume confirmation
# Long when price breaks above Camarilla R3 in bullish regime (ADX>25) with volume spike
# Short when price breaks below Camarilla S3 in bearish regime (ADX>25) with volume spike
# Uses 1d ADX to filter for trending markets only, avoiding whipsaws in ranging conditions
# Volume confirmation ensures breakouts have institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe

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
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R3 = Close + 1.1*(High-Low)*1.1/4
    # S3 = Close - 1.1*(High-Low)*1.1/4
    camarilla_window = 1
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r3_level = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3_level = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for ADX and Camarilla
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        is_trending = curr_adx > 25
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in trending regime
            if is_trending and curr_volume_confirm:
                # Bullish breakout: price breaks above Camarilla R3
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Camarilla S3
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to midpoint between R3 and S3 OR breaks below S3 with volume
            midpoint = (curr_r3 + curr_s3) / 2.0
            
            if curr_close <= midpoint or (curr_close < curr_s3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to midpoint between R3 and S3 OR breaks above R3 with volume
            midpoint = (curr_r3 + curr_s3) / 2.0
            
            if curr_close >= midpoint or (curr_close > curr_r3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals