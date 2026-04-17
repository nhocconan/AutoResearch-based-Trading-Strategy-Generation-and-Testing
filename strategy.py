#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h volume filter and 1d volatility regime.
Long when 1h close breaks above R3 and 4h volume > 1.5x 20-period average and 1d ATR ratio < 0.8 (low volatility).
Short when 1h close breaks below S3 and 4h volume > 1.5x 20-period average and 1d ATR ratio < 0.8.
Exit when price returns to Camarilla R1/S1 levels or 1d ATR ratio > 1.2 (high volatility).
Uses 4h for volume confirmation, 1d for volatility regime filter, 1h for precise entry timing.
Designed to capture breakouts in low volatility environments which work in both bull and bear markets.
Target: 20-40 trades/year per symbol to minimize fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    
    # Get 4h data for volume filter
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h volume MA20
    volume_4h_series = pd.Series(volume_4h)
    vol_ma_20_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) and ATR ratio (current ATR / 20-period average ATR)
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_1d / atr_ma_20_1d, np.nan)
    
    # Align 4h volume MA and 1d ATR ratio to 1h timeframe
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 1h Camarilla pivot levels (based on previous day's OHLC)
    # We need to get daily OHLC aligned to 1h timeframe
    df_1d_ohlc = get_htf_data(prices, '1d')
    open_1d = df_1d_ohlc['open'].values
    high_1d_ohlc = df_1d_ohlc['high'].values
    low_1d_ohlc = df_1d_ohlc['low'].values
    close_1d_ohlc = df_1d_ohlc['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.full_like(close_1d_ohlc, np.nan)
    camarilla_r2 = np.full_like(close_1d_ohlc, np.nan)
    camarilla_r1 = np.full_like(close_1d_ohlc, np.nan)
    camarilla_s1 = np.full_like(close_1d_ohlc, np.nan)
    camarilla_s2 = np.full_like(close_1d_ohlc, np.nan)
    camarilla_s3 = np.full_like(close_1d_ohlc, np.nan)
    
    # Camarilla formulas: 
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # We'll use R3, R2, R1, S1, S2, S3
    range_1d = high_1d_ohlc - low_1d_ohlc
    camarilla_r3 = close_1d_ohlc + (range_1d * 1.1 / 4)
    camarilla_r2 = close_1d_ohlc + (range_1d * 1.1 / 6)
    camarilla_r1 = close_1d_ohlc + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d_ohlc - (range_1d * 1.1 / 12)
    camarilla_s2 = close_1d_ohlc - (range_1d * 1.1 / 6)
    camarilla_s3 = close_1d_ohlc - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (using previous day's values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period average
        # Get the aligned 4h volume for this timestamp
        volume_4h_full = df_4h['volume'].values
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h_full)
        volume_confirmed = not np.isnan(volume_4h_aligned[i]) and \
                          not np.isnan(vol_ma_20_4h_aligned[i]) and \
                          volume_4h_aligned[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        # Volatility regime filter: 1d ATR ratio < 0.8 (low volatility environment)
        vol_regime = not np.isnan(atr_ratio_1d_aligned[i]) and atr_ratio_1d_aligned[i] < 0.8
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        # Reversion conditions (exit signals)
        revert_to_r1 = close[i] < camarilla_r1_aligned[i]
        revert_to_s1 = close[i] > camarilla_s1_aligned[i]
        
        # High volatility exit condition
        high_vol_exit = not np.isnan(atr_ratio_1d_aligned[i]) and atr_ratio_1d_aligned[i] > 1.2
        
        if position == 0:
            # Long: breakout above R3 with volume confirmation and low volatility regime
            if (breakout_up and volume_confirmed and vol_regime):
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3 with volume confirmation and low volatility regime
            elif (breakout_down and volume_confirmed and vol_regime):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to R1 OR high volatility environment
            if (revert_to_r1 or high_vol_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to S1 OR high volatility environment
            if (revert_to_s1 or high_vol_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hVolume_1dATRRegime"
timeframe = "1h"
leverage = 1.0