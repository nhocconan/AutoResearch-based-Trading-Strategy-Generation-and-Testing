#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_ATRStop
Hypothesis: 4h Camarilla pivot (R1/S1) breakouts filtered by 1d EMA50 trend, volume regime (low chop = trending), and ATR trailing stop.
Enter long when price breaks above 4h R1 with 1d uptrend and low chop regime (trending market).
Enter short when price breaks below 4h S1 with 1d downtrend and low chop regime.
Exit on ATR(14) trailing stop (2.5*ATR) or opposite level break.
Designed for low trade frequency (~20-40 trades/year) to minimize fee drag.
Uses chop regime filter to avoid false breakouts in ranging markets, improving performance in both bull and bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for pivots, 1d for trend and chop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h Camarilla Pivot Levels (R1, S1) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_4h - low_4h) * 1.1 / 12.0
    r1_4h = close_4h + camarilla_range
    s1_4h = close_4h - camarilla_range
    
    # Align to 4h timeframe (use previous completed 4h bar)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 1d EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (log(n) * (highest_high - lowest_low))) over n periods
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_1d = highest_high_1d - lowest_low_1d
    
    # Avoid division by zero
    log_n = np.log10(14)  # constant for 14-period
    chop_denominator = log_n * range_1d
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_value = 100 * np.log10(sum_atr_1d / chop_denominator)
    
    # Align chop to LTF
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    # Chop regime: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = ranging (favor mean reversion)
    # We want trending regime for breakouts: CHOP < 38.2
    chop_regime = chop_1d_aligned < 38.2
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) 
            or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 1.5x 20-period average (less strict than 2x to allow more trades)
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 4h R1, 1d uptrend, low chop (trending), volume spike
            long_breakout = price > r1_4h_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            long_chop = chop_regime[i]  # CHOP < 38.2 = trending
            
            # Short conditions: price < 4h S1, 1d downtrend, low chop (trending), volume spike
            short_breakout = price < s1_4h_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            short_chop = chop_regime[i]  # CHOP < 38.2 = trending
            
            # Entry logic
            if long_breakout and long_trend and long_chop and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and short_chop and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 4h S1 (support broken)
            elif price < s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 4h R1 (resistance broken)
            elif price > r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_ATRStop"
timeframe = "4h"
leverage = 1.0