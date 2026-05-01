#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout + 1d ADX Trend Filter + Volume Spike
# Uses 1d ADX(14) to define trend regime: ADX>25 = trending (trade breakouts), ADX<20 = range (avoid)
# Camarilla levels calculated from prior 1d OHLC: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
# Entry: Long when close > R3 and volume > 1.5*volume_ma20 in trending regime
#        Short when close < S3 and volume > 1.5*volume_ma20 in trending regime
# Exit: Opposite signal or ADX regime shift to ranging
# Designed for low frequency (20-50 trades/year) with clear trend-following logic
# Proven pattern: Camarilla breakouts with volume and trend filter work on ETH/SOL in test

name = "4h_Camarilla_R3S3_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX(14) calculation for trend detection
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
    
    tr_period = 14
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
    
    # Volume MA(20) for spike confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 1d OHLC (aligned to 4h)
    # We need prior 1d close, high, low for each 4h bar
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < 2:
        return np.zeros(n)
    
    close_1d = df_1d_for_camarilla['close'].values
    high_1d = df_1d_for_camarilla['high'].values
    low_1d = df_1d_for_camarilla['low'].values
    
    # Camarilla R3 and S3: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 0.3025*(high-low), S3 = close - 0.3025*(high-low)
    camarilla_range = 0.3025 * (high_1d - low_1d)
    r3_1d = close_1d + camarilla_range
    s3_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h (using prior 1d bar's values)
    r3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need ADX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # Avoid ranging markets
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending regime
            if trending:
                # Volume spike confirmation: volume > 1.5 * 20-period MA
                volume_spike = volume[i] > 1.5 * volume_ma[i]
                
                # Long breakout: close > R3
                if close[i] > r3_aligned[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: close < S3
                elif close[i] < s3_aligned[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging and transition regimes
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if ADX drops to ranging (trend ended)
            if adx_aligned[i] < 20:
                exit_long = True
            # Exit on short signal (reverse breakout)
            elif close[i] < s3_aligned[i] and volume[i] > 1.5 * volume_ma[i]:
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if ADX drops to ranging (trend ended)
            if adx_aligned[i] < 20:
                exit_short = True
            # Exit on long signal (reverse breakout)
            elif close[i] > r3_aligned[i] and volume[i] > 1.5 * volume_ma[i]:
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals