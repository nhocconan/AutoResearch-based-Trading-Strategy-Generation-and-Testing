#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter, volume spike, and choppiness regime filter.
Long when price breaks above R3 with 1d uptrend, volume spike, and chop < 38.2 (trending market).
Short when price breaks below S3 with 1d downtrend, volume spike, and chop < 38.2.
Uses tighter R3/S3 levels for fewer, higher-quality breakouts. Volume spike confirms institutional participation.
1d trend filter ensures alignment with higher timeframe momentum. Chop filter avoids false breakouts in ranging markets.
Designed for 15-30 trades/year on 4h to minimize fee drag while capturing strong directional moves in both bull and bear markets.
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
    
    # Calculate 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for chop filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate True Range sum for chop filter (Choppiness Index components)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Calculate highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop = np.where(hl_range > 0, 100 * np.log10(tr_sum / hl_range) / np.log10(14), 50)
    
    # Calculate Camarilla levels for each 4h bar using previous 4h bar's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low),
    # R2 = close + 0.75*(high-low), R1 = close + 0.5*(high-low),
    # S1 = close - 0.5*(high-low), S2 = close - 0.75*(high-low),
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 for breakout signals (tighter than R1/S1)
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    camarilla_range = high_shift - low_shift
    r3 = close_shift + 1.125 * camarilla_range
    s3 = close_shift - 1.125 * camarilla_range
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for previous bar data, EMA34, ATR, volume average, chop
    start_idx = max(34, 14, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla R3/S3 breakout with 1d trend alignment, volume spike, and low chop (trending market)
            # Long: Close > R3 AND 1d trend up (close > EMA34) AND volume spike AND chop < 38.2 (trending)
            # Short: Close < S3 AND 1d trend down (close < EMA34) AND volume spike AND chop < 38.2 (trending)
            long_condition = (close_val > r3[i] and 
                            close_val > ema_trend and 
                            vol_spike and 
                            chop_val < 38.2)
            short_condition = (close_val < s3[i] and 
                             close_val < ema_trend and 
                             vol_spike and 
                             chop_val < 38.2)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below S3 (reversal) OR 1d trend turns down OR chop becomes high (ranging)
            if close_val < s3[i] or close_val < ema_trend or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R3 (reversal) OR 1d trend turns up OR chop becomes high (ranging)
            if close_val > r3[i] or close_val > ema_trend or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0