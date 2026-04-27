#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_Regime
Hypothesis: Camarilla R1/S1 levels from 1d chart act as intraday support/resistance. 
Breakout above R1 with volume confirmation and 1d uptrend (price > EMA34) goes long.
Breakdown below S1 with volume confirmation and 1d downtrend (price < EMA34) goes short.
Adds choppiness regime filter: only trade when CHOP(14) < 61.8 (trending market).
Exits on opposite Camarilla level touch or trend reversal. Uses discrete sizing (0.25) 
to limit fee churn. Designed for 4h timeframe targeting 75-200 total trades over 4 years 
(19-50/year). Works in bull markets via upside breakouts and bear markets via 
downside breakdowns, with volume and regime filters preventing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rang = high_1d - low_1d
    R1 = close_1d + rang * 1.1 / 2
    S1 = close_1d - rang * 1.1 / 2
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d choppiness index: CHOP(14) = 100 * log10(sum(ATR(14)) / log10(range(14)))
    # Simplified: CHOP = 100 * log10(sum(TR(14)) / log10(highest_high - lowest_low over 14))
    tr1 = np.maximum(high_1d[1:] - low_1d[:-1], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range14 = highest_high14 - lowest_low14
    chop = 100 * np.log10(sum_atr14) / np.log10(range14)
    chop = np.where(range14 == 0, 50, chop)  # avoid division by zero
    
    # Align all indicators to primary timeframe (4h)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA34 (34), volume avg (20), chop (14), ATR (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        ema_1d_val = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Regime filter: only trade when market is trending (CHOP < 61.8)
        if chop_val >= 61.8:
            # Choppy market: exit any position, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine 1d trend: price > EMA34 = uptrend, price < EMA34 = downtrend
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R1 and volume confirms
                if (close_val > R1_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S1 and volume confirms
                if (close_val < S1_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S1 (support) or trend changes to downtrend
            exit_condition = (close_val < S1_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R1 (resistance) or trend changes to uptrend
            exit_condition = (close_val > R1_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_Regime"
timeframe = "4h"
leverage = 1.0