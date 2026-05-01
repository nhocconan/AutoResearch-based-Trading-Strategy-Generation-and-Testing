#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1d ADX Trend Filter and Volume Spike
# Uses 1d ADX(14) to confirm trending regime (ADX>25) and 12h volume spike (1.5x 20-bar avg)
# Enters long when price breaks above Camarilla R3, short when breaks below S3
# Exits on opposite Camarilla level (R4/S4) or ADX<20 (range regime)
# Designed for low frequency (50-150 trades over 4 years) with clear trend following logic

name = "12h_Camarilla_R3S3_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d ADX(34) calculation for stronger trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_period = 34
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h Camarilla levels (based on previous day's OHLC)
    # Calculate daily OHLC from 1d data
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + (rng * 1.1 / 4)
    camarilla_s3 = prev_close - (rng * 1.1 / 4)
    camarilla_r4 = prev_close + (rng * 1.1 / 2)
    camarilla_s4 = prev_close - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 12h volume spike filter (1.5x 20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need ADX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX>25)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if trending and volume_spike[i]:
                # Long: price breaks above Camarilla R3
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Camarilla S3
                elif close[i] < camarilla_s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No trade in ranging regime or no volume spike
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on opposite Camarilla level (R4) or ADX<20 (range regime)
            if close[i] >= camarilla_r4_aligned[i]:
                exit_long = True
            elif adx_aligned[i] < 20:
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on opposite Camarilla level (S4) or ADX<20 (range regime)
            if close[i] <= camarilla_s4_aligned[i]:
                exit_short = True
            elif adx_aligned[i] < 20:
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals