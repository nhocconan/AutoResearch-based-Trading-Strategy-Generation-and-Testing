#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm
Hypothesis: Camarilla R3/S3 breakouts on 6h with 1d EMA34 trend filter, volume spike, and chop regime filter. Uses tighter entry (R3/S3) and higher timeframe confirmation (1d trend + chop) to reduce trades to 50-150 over 4 years while maintaining edge in both bull/bear markets via 1d trend filter. Position size 0.25 balances return and drawdown.
Primary timeframe: 6h, HTF: 1d for trend and Camarilla calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R3, S3)
    # Camarilla: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First day: use same values (will be filtered by min_periods later)
    high_1d_prev[0] = high_1d[0]
    low_1d_prev[0] = low_1d[0]
    close_1d_prev[0] = close_1d[0]
    
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    camarilla_range = high_1d_prev - low_1d_prev
    r3 = close_1d_prev + camarilla_range * 1.1 / 4
    s3 = close_1d_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 6h (no extra delay needed as they're based on completed 1d candles)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: volume > 2.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Choppiness regime filter: CHOP < 50 = trending market (favor breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_period = 14
    chop_period = 14
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # first TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values / price_range) / np.log10(chop_period)
    chop_filter = chop < 50  # trending market
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 20 for volume median, 14 for ATR/CHOP
    start_idx = max(34, 20, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R3 with volume spike, uptrend (close > EMA34_1d), and trending market
            long_entry = (close_val > r3_aligned[i]) and vol_spike and (close_val > ema_34_val) and chop_ok
            # Short: price breaks below S3 with volume spike, downtrend (close < EMA34_1d), and trending market
            short_entry = (close_val < s3_aligned[i]) and vol_spike and (close_val < ema_34_val) and chop_ok
            
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
            # Long - exit on trend reversal or price re-enters Camarilla (below S3)
            if close_val < ema_34_val or close_val < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price re-enters Camarilla (above R3)
            if close_val > ema_34_val or close_val > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTFConfirm"
timeframe = "6h"
leverage = 1.0