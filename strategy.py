#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX Regime Filter and Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for regime filter (ADX > 25 = trending, ADX < 20 = range).
- Bull Power = High - EMA13(close); Bear Power = Low - EMA13(close).
- Entry: Long when Bull Power > 0 AND ADX > 25 (strong uptrend) AND volume > 2.0 * 6h volume MA(20);
         Short when Bear Power < 0 AND ADX > 25 (strong downtrend) AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when Bull Power <= 0; Short exits when Bear Power >= 0.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying strength in uptrend) and bear (selling weakness in downtrend) with volume confirmation to avoid false signals.
- Uses Elder Ray to measure price strength relative to EMA, ADX to filter regimes, and volume to confirm conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate EMA13 for 6h data (for Elder Ray)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Get 6h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 28)  # EMA13 needs 13, volume MA needs 20, ADX needs ~28 (14+14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and adx_aligned[i] > 25:
                # Long: Bull Power > 0 (price above EMA13) AND strong uptrend (ADX > 25)
                if bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (price below EMA13) AND strong downtrend (ADX > 25)
                elif bear_power[i] < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Bull Power <= 0 (price falls below EMA13)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bear Power >= 0 (price rises above EMA13)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0