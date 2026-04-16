#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike and chop regime filter
# Long when price breaks above 4h Camarilla R1 AND 1d volume > 1.5x 20-period 1d volume SMA AND 1d chop > 61.8 (ranging market)
# Short when price breaks below 4h Camarilla S1 AND 1d volume > 1.5x 20-period 1d volume SMA AND 1d chop > 61.8 (ranging market)
# Camarilla pivots identify intraday support/resistance, volume confirms conviction, chop filter avoids whipsaws in strong trends
# Discrete position sizing (0.25) to control drawdown. Target: 75-200 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data once before loop for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: Camarilla Pivot Points (based on previous day) ===
    # Calculate using previous 1d OHLC, aligned to 4h
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1d Indicator: Choppy Market Index (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppy Market Index: 100 * log10(sum(tr)/(hh_ll)) / log10(14)
    hh_ll = hh_14 - ll_14
    chop = np.where(hh_ll > 0, 100 * np.log10(tr_sum / hh_ll) / np.log10(14), 50)
    chop = np.where((tr_sum == 0) | (hh_ll == 0), 50, chop)
    
    # Align chop to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (need 20 periods for vol SMA + 14 for chop)
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        vol_confirm = False
        if not np.isnan(vol_1d_aligned[i]):
            vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
            vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Chop filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        chop_filter = chop_aligned[i] > 61.8
        
        # === LONG CONDITIONS ===
        # Price breaks above 4h Camarilla R1 AND volume confirmation AND chop filter
        if (close[i] > camarilla_r1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below 4h Camarilla S1 AND volume confirmation AND chop filter
        elif (close[i] < camarilla_s1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1_S1_1dVolSpike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0