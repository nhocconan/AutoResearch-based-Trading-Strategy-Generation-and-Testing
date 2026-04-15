#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 4h Camarilla R3 + 1d volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Short when price breaks below 4h Camarilla S3 + 1d volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# Camarilla levels provide mean-reversion breakout points in ranging markets. Volume confirms participation.
# Choppiness filter ensures we only trade in ranging regimes where Camarilla works best.
# Works in bull markets (buying dips to R3) and bear markets (selling rallies to S3) by requiring chop > 61.8.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    vol_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1d Indicator: Choppiness Index (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM (for ADX component of Choppiness)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    plus_dm = np.where((high_1d - high_1d_shift) > (low_1d_shift - low_1d), 
                       np.maximum(high_1d - high_1d_shift, 0), 0)
    minus_dm = np.where((low_1d_shift - low_1d) > (high_1d - high_1d_shift), 
                        np.maximum(low_1d_shift - low_1d, 0), 0)
    
    # Wilder's smoothing for TR, +DM, -DM
    period = 14
    alpha = 1.0 / period
    
    atr_1d = np.zeros_like(tr)
    atr_1d[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Calculate +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di_1d = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
    
    # Calculate DX and ADX (for Choppiness denominator)
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    
    adx_1d = np.zeros_like(dx_1d)
    adx_1d[2*period-1] = np.mean(dx_1d[period-1:2*period])
    for i in range(2*period, len(dx_1d)):
        adx_1d[i] = (adx_1d[i-1] * (period-1) + dx_1d[i]) / period
    
    # Choppiness Index = 100 * log10(sum(TR)/sum(ATR)) / log10(period)
    # Simplified: CHOP = 100 * log10(period * ATR / (period * TR)) / log10(period) 
    # Actually: CHOP = 100 * log10(sum(TR over period) / sum(ATR over period)) / log10(period)
    # We'll use the standard formula: CHOP = 100 * log10(sum(TR) / sum(ATR)) / log10(N)
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    sum_atr = pd.Series(atr_1d).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero and log of zero
    chop_ratio = np.where((sum_atr > 0) & (sum_tr > 0), sum_tr / sum_atr, 1.0)
    chop_1d = np.where(chop_ratio > 0, 100 * np.log10(chop_ratio) / np.log10(period), 50.0)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicator: Camarilla Pivot Levels (R3, S3) ===
    # Based on previous day's OHLC
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    
    # We need daily OHLC for 4h calculation - use 1d data shifted by 1
    # For each 4h bar, we use the previous completed 1d bar's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 2*period) + 20  # volume SMA(20) + CHOP(28) + Camarilla (need prev day)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA (from 1d)
        vol_confirm = volume[i] > (vol_sma_20_1d_aligned[i] * 2.0)
        
        # Chop filter: CHOP > 61.8 (ranging market)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Camarilla R3
        # 2. Volume confirmation
        # 3. Choppiness filter (ranging market)
        if (close[i] > camarilla_r3_4h_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Camarilla S3
        # 2. Volume confirmation
        # 3. Choppiness filter (ranging market)
        elif (close[i] < camarilla_s3_4h_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_1dVolume2x_CHOP_Filter_v1"
timeframe = "4h"
leverage = 1.0