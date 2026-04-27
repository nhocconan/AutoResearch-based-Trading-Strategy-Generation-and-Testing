#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA34 trend filter, choppiness index regime (CHOP > 61.8 = range, < 38.2 = trend), and volume spike confirmation (>2.5x average). Uses discrete 0.30 position size to limit fee drift. Designed to work in both bull and bear: trend filter ensures alignment with higher timeframe, chop regime avoids whipsaw in sideways markets, volume confirms genuine breakout participation. Target: 80-150 trades over 4 years (20-38/year).
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.08)   # R1 level
    s1 = prev_close - (rng * 1.08)   # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    # Choppiness Index regime filter: CHOP(14) < 38.2 = trending (favor trend following)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * sqrt(14)))
    # Simplified: use ATR(14) and price range over 14 periods
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    price_range = highest_high - lowest_low
    chop = 100 * np.log10(sum_atr14 / (np.log10(price_range) * np.sqrt(14)))
    # Regime: CHOP < 38.2 = trending (favor), CHOP > 61.8 = ranging (avoid)
    chop_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.30   # Fixed position size to minimize churn
    
    # Warmup: need 1d EMA34 (34), 1d shift(1) for Camarilla, ATR14 (14), sum ATR14 (14), HH/LL (14)
    start_idx = max(34 + 1, 1 + 1, 14, 14 + 14, 14 + 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        chop_reg = chop_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1d EMA34 alignment, volume confirmation, and chop regime (trending)
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            chop_reg)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             chop_reg)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0