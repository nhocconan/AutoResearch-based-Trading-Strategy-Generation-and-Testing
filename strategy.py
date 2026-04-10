#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot mean reversion with 1w trend filter and volume confirmation
# - Camarilla levels from 1d: mean reversion at S3/R3 (extreme levels) in ranging markets
# - 1w ADX(14) < 20 to ensure ranging market and avoid strong trends
# - Volume confirmation: current 1d volume > 1.5x 20-period average to validate reversal
# - Designed for 1d timeframe: targets 30-100 trades over 4 years (7-25/year) to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we trade only in ranging conditions
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "1d_1w_camarilla_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for trend filter (ranging market when ADX < 20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Calculate pivot from previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current values
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    S3 = pivot - (range_hl * 1.1 / 4.0)
    S2 = pivot - (range_hl * 1.1 / 6.0)
    S1 = pivot - (range_hl * 1.1 / 12.0)
    R1 = pivot + (range_hl * 1.1 / 12.0)
    R2 = pivot + (range_hl * 1.1 / 6.0)
    R3 = pivot + (range_hl * 1.1 / 4.0)
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start from 20 to ensure volume average is valid
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(S3[i]) or np.isnan(R3[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches midpoint (mean reversion complete) or stop loss
            if close_1d[i] >= pivot[i] or close_1d[i] < S3[i] - 0.5 * (pivot[i] - S3[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches midpoint (mean reversion complete) or stop loss
            if close_1d[i] <= pivot[i] or close_1d[i] > R3[i] + 0.5 * (R3[i] - pivot[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion setup in ranging market
            if vol_spike[i] and adx_aligned[i] < 20:
                # Mean reversion long: price at S3 extreme with rejection
                if close_1d[i] <= S3[i] and close_1d[i] > S3[i] - 0.3 * (pivot[i] - S3[i]):
                    position = 1
                    signals[i] = 0.25
                # Mean reversion short: price at R3 extreme with rejection
                elif close_1d[i] >= R3[i] and close_1d[i] < R3[i] + 0.3 * (R3[i] - pivot[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals