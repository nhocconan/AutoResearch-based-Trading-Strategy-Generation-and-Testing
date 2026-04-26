#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_ChopFilter_VolumeSpike
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 level AND 1d trend is up (close > EMA34) AND chop regime is trending (CHOP < 40) AND volume > 2.5x 20-period average. Enter short when price breaks below S3 level AND 1d trend is down (close < EMA34) AND chop regime is trending (CHOP < 40) AND volume spike. Uses Camarilla pivot levels for precise S/R, 1d EMA34 for higher timeframe trend alignment, Choppiness Index for regime filter to avoid whipsaws in ranging markets, and volume confirmation for institutional participation. Designed for low trade frequency (15-25/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Daily Camarilla Pivot Levels (R3, S3)
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + ((high-low)*1.1/4), S3 = close - ((high-low)*1.1/4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate Choppiness Index on 1d timeframe for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Where ATR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar has no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    atr_14 = np.zeros_like(tr1)
    for i in range(len(tr1)):
        if i < 14:
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.nanmean(np.column_stack([tr1[i-13:i+1], tr2[i-13:i+1], tr3[i-13:i+1]]))
    
    # Choppiness Index
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if np.isnan(atr_14[i]):
            chop[i] = np.nan
        else:
            atr_sum = np.nansum(atr_14[i-13:i+1])
            if atr_sum > 0 and close_1d[i] != close_1d[i-14]:
                chop[i] = 100 * np.log10(atr_sum) / np.log10(14)
            else:
                chop[i] = 50.0  # neutral when undefined
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 2.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (34), ATR warmup (14+14=28), volume MA warmup (20)
    start_idx = max(34, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 40)
        trending_regime = chop_aligned[i] < 40.0
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above R3 + 1d uptrend + trending regime + volume spike
            long_signal = breakout_above_r3 and trend_uptrend and trending_regime and volume_spike[i]
            
            # Short: price below S3 + 1d downtrend + trending regime + volume spike
            short_signal = breakout_below_s3 and trend_downtrend and trending_regime and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 OR trend change to downtrend OR regime becomes ranging
            if (close[i] < camarilla_s3_aligned[i] or not trend_uptrend or chop_aligned[i] >= 50.0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR trend change to uptrend OR regime becomes ranging
            if (close[i] > camarilla_r3_aligned[i] or not trend_downtrend or chop_aligned[i] >= 50.0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_ChopFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0