#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Parabolic SAR with 12h ADX for trend strength.
# Long when price crosses above Parabolic SAR and ADX > 25 (trending).
# Short when price crosses below Parabolic SAR and ADX > 25.
# Exit when price crosses back through Parabolic SAR.
# Uses 1d Parabolic SAR to capture multi-day trends, filtered by 12h ADX to avoid chop.
# Designed for low trade frequency (<30/year) to avoid fee drag.

name = "4h_1dPSAR_12hADX_Trend"
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
    
    # Get 1d data for Parabolic SAR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.20
    
    psar = np.zeros_like(close_1d)
    psar_up = np.zeros_like(close_1d)
    psar_down = np.zeros_like(close_1d)
    bull = np.ones_like(close_1d, dtype=bool)
    af = np.full_like(close_1d, af_start)
    ep = np.zeros_like(close_1d)
    
    # Initialize
    psar[0] = low_1d[0]
    ep[0] = high_1d[0]
    
    for i in range(1, len(close_1d)):
        if bull[i-1]:
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if low_1d[i] < psar[i]:
                bull[i] = False
                psar[i] = ep[i-1]
                ep[i] = low_1d[i]
                af[i] = af_start
            else:
                bull[i] = True
                if high_1d[i] > ep[i-1]:
                    ep[i] = high_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if high_1d[i] > psar[i]:
                bull[i] = True
                psar[i] = ep[i-1]
                ep[i] = high_1d[i]
                af[i] = af_start
            else:
                bull[i] = False
                if low_1d[i] < ep[i-1]:
                    ep[i] = low_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Get 12h data for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth with Wilder's method (period=14)
    period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Initial values
    if len(tr) >= period + 1:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    mask = ~np.isnan(atr) & (atr != 0)
    di_plus[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
    di_minus[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    dx_mask = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    adx = np.full_like(tr, np.nan)
    if len(tr) >= 2 * period + 1:
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period + 1, len(tr)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Align indicators to 4h timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(psar_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above PSAR and ADX > 25 with volume confirmation
            if (close[i] > psar_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below PSAR and ADX > 25 with volume confirmation
            elif (close[i] < psar_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below PSAR
            if close[i] < psar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above PSAR
            if close[i] > psar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals