#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily ATR breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above daily ATR-based upper band with volume surge and ADX > 25.
# Short when price breaks below daily ATR-based lower band with volume surge and ADX > 25.
# Uses 1d ATR and ADX to capture volatility expansion in trending markets.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Works in both bull and bear markets by following trends.

name = "12h_1dATRBreakout_ADXTrend"
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
    
    # Get 1d data for ATR and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily data
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(14, len(tr)):
        if np.isnan(atr_14[i-1]):
            atr_14[i] = np.nanmean(tr[i-13:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate ADX (14-period) on daily data
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, TR
    tr_plus = np.full_like(high_1d, np.nan)
    tr_minus = np.full_like(high_1d, np.nan)
    tr_tr = np.full_like(high_1d, np.nan)
    
    for i in range(14, len(high_1d)):
        if np.isnan(tr_plus[i-1]):
            tr_plus[i] = np.nansum(plus_dm[i-13:i+1])
            tr_minus[i] = np.nansum(minus_dm[i-13:i+1])
            tr_tr[i] = np.nansum(tr[i-13:i+1])
        else:
            tr_plus[i] = tr_plus[i-1] - (tr_plus[i-1] / 14) + plus_dm[i]
            tr_minus[i] = tr_minus[i-1] - (tr_minus[i-1] / 14) + minus_dm[i]
            tr_tr[i] = tr_tr[i-1] - (tr_tr[i-1] / 14) + tr[i]
    
    # DI+ and DI-
    plus_di = np.full_like(high_1d, np.nan)
    minus_di = np.full_like(high_1d, np.nan)
    for i in range(14, len(high_1d)):
        if tr_tr[i] != 0:
            plus_di[i] = 100 * tr_plus[i] / tr_tr[i]
            minus_di[i] = 100 * tr_minus[i] / tr_tr[i]
    
    # DX and ADX
    dx = np.full_like(high_1d, np.nan)
    for i in range(14, len(high_1d)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx_14 = np.full_like(high_1d, np.nan)
    for i in range(28, len(dx)):  # ADX needs 2*period
        if np.isnan(adx_14[i-1]):
            adx_14[i] = np.nanmean(dx[i-13:i+1])
        else:
            adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
    
    # Calculate daily ATR-based bands (using previous day's ATR)
    upper_band = np.full_like(close_1d, np.nan)
    lower_band = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        if not np.isnan(atr_14[i-1]):
            upper_band[i] = close_1d[i-1] + 1.5 * atr_14[i-1]
            lower_band[i] = close_1d[i-1] - 1.5 * atr_14[i-1]
    
    # For first day, use close price
    if len(close_1d) >= 1:
        upper_band[0] = close_1d[0]
        lower_band[0] = close_1d[0]
    
    # Align 1d indicators to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: 12h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(adx_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band + volume surge + ADX > 25
            if close[i] > upper_band_aligned[i] and vol_spike[i] and adx_14_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band + volume surge + ADX > 25
            elif close[i] < lower_band_aligned[i] and vol_spike[i] and adx_14_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band
            if close[i] < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band
            if close[i] > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals