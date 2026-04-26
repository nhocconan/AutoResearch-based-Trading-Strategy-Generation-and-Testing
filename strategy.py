#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d EMA50 trend filter, volume confirmation, and chop regime filter. 
Only trades when market is trending (CHOP < 38.2) to avoid whipsaws in ranging markets. 
Targets 20-50 trades/year by requiring confluence of trend, volume, regime, and precise breakout levels. 
Works in bull/bear markets via 1d trend filter and regime adaptation.
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
    
    # Load 1d data ONCE before loop for HTF trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla pivot levels for 1d (using previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 for breakouts
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range
    s3 = prev_close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate Choppiness Index on 1d for regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Simplified: use rolling ATR sum over 14 periods
    atr_14_for_chop = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_denominator = np.log10(14)
    chop_value = 100 * (np.log10(atr_14_for_chop) / chop_denominator)
    chop_value = np.where(chop_denominator > 0, chop_value, 50)  # default to neutral if invalid
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Reduced fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 14 for ATR, 20 for volume median, 1 for chop
    start_idx = max(50, 14, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Regime filter: only trade when trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Flat - look for entry
            # Bullish breakout: close > R3 and volume spike, in uptrend (close > EMA50_1d), and trending regime
            long_entry = (close_val > r3_val) and vol_spike and (close_val > ema_50_val) and is_trending
            # Bearish breakout: close < S3 and volume spike, in downtrend (close < EMA50_1d), and trending regime
            short_entry = (close_val < s3_val) and vol_spike and (close_val < ema_50_val) and is_trending
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, ATR stoploss, or close below S3 (take profit)
            stop_price = entry_price - 2.0 * atr_val
            if (close_val < ema_50_val or 
                close_val < stop_price or 
                close_val < s3_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, ATR stoploss, or close above R3 (take profit)
            stop_price = entry_price + 2.0 * atr_val
            if (close_val > ema_50_val or 
                close_val > stop_price or 
                close_val > r3_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0