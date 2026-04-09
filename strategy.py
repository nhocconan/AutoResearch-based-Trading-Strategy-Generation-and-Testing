#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly trend filter and volume confirmation
# Uses weekly ADX to filter trend direction, daily Donchian(20) breakout for entry
# Volume > 1.5x 20-period average confirms breakout strength
# Fixed position size 0.25 to limit drawdown and control risk
# Designed for ~15-25 trades/year (~60-100 total over 4 years) to minimize fee drag
# Works in bull markets (breakouts continuation) and bear markets (false breakouts filtered by weekly trend)

name = "1d_donchian_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    wn = len(wh)
    
    # True Range
    wtr = np.full(wn, np.nan)
    wdm_plus = np.full(wn, np.nan)
    wdm_minus = np.full(wn, np.nan)
    
    for i in range(1, wn):
        wtr0 = wh[i] - wl[i]
        wtr1 = abs(wh[i] - wc[i-1])
        wtr2 = abs(wl[i] - wc[i-1])
        wtr[i] = max(wtr0, wtr1, wtr2)
        
        wup = wh[i] - wh[i-1]
        wdown = wl[i-1] - wl[i]
        if wup > wdown and wup > 0:
            wdm_plus[i] = wup
        else:
            wdm_plus[i] = 0.0
        if wdown > wup and wdown > 0:
            wdm_minus[i] = wdown
        else:
            wdm_minus[i] = 0.0
    
    # Smoothed averages (Wilder smoothing)
    wtr14 = np.full(wn, np.nan)
    wdm_plus_14 = np.full(wn, np.nan)
    wdm_minus_14 = np.full(wn, np.nan)
    
    if wn >= 14:
        wtr14[13] = np.nansum(wtr[1:14])
        wdm_plus_14[13] = np.nansum(wdm_plus[1:14])
        wdm_minus_14[13] = np.nansum(wdm_minus[1:14])
        
        for i in range(14, wn):
            wtr14[i] = wtr14[i-1] - (wtr14[i-1] / 14) + wtr[i]
            wdm_plus_14[i] = wdm_plus_14[i-1] - (wdm_plus_14[i-1] / 14) + wdm_plus[i]
            wdm_minus_14[i] = wdm_minus_14[i-1] - (wdm_minus_14[i-1] / 14) + wdm_minus[i]
    
    # DI and DX
    wdi_plus = np.full(wn, np.nan)
    wdi_minus = np.full(wn, np.nan)
    wdx = np.full(wn, np.nan)
    
    for i in range(14, wn):
        if wtr14[i] > 0:
            wdi_plus[i] = 100 * wdm_plus_14[i] / wtr14[i]
            wdi_minus[i] = 100 * wdm_minus_14[i] / wtr14[i]
            wdx[i] = 100 * abs(wdi_plus[i] - wdi_minus[i]) / (wdi_plus[i] + wdi_minus[i])
    
    # ADX (smoothed DX)
    wadx = np.full(wn, np.nan)
    if wn >= 28:
        wadx[27] = np.nansum(wdx[14:28]) / 14
        for i in range(28, wn):
            wadx[i] = (wadx[i-1] * 13 + wdx[i]) / 14
    
    # Align weekly ADX to daily timeframe
    wadx_daily = align_htf_to_ltf(prices, df_1w, wadx)
    
    # Daily Donchian channel (20-period)
    dcm = np.full(n, np.nan)  # upper band
    dcl = np.full(n, np.nan)  # lower band
    
    for i in range(n):
        if i >= 19:
            dcm[i] = np.max(high[i-19:i+1])
            dcl[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(dcm[i]) or 
            np.isnan(dcl[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(wadx_daily[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR weekly ADX < 20 (weak trend)
            if close[i] < dcl[i] or wadx_daily[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR weekly ADX < 20
            if close[i] > dcm[i] or wadx_daily[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian upper band with volume confirmation AND weekly ADX > 25
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > dcm[i] and 
                vol_ratio > 1.5 and 
                wadx_daily[i] > 25):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower band with volume confirmation AND weekly ADX > 25
            elif (close[i] < dcl[i] and 
                  vol_ratio > 1.5 and 
                  wadx_daily[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals