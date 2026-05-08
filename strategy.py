#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d trend filter and volume spike confirmation.
# Long when price touches or breaks below Camarilla S1 (support) AND 1d EMA34 rising AND volume > 1.8x 20-period average.
# Short when price touches or breaks above Camarilla R1 (resistance) AND 1d EMA34 falling AND volume > 1.8x 20-period average.
# Exit when price reverses back to Camarilla PP (pivot point) or closes beyond opposite S/R level.
# This strategy captures mean-reversion bounces at key intraday levels with trend alignment and volume confirmation.
# Camarilla levels provide high-probability reversal zones. The 1d EMA34 filter ensures we trade with the higher timeframe trend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R1_S1_1dEMA34_Volume"
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
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels calculation
    # R4 = Close + ((High-Low) * 1.5/2)
    # R3 = Close + ((High-Low) * 1.25/2)
    # R2 = Close + ((High-Low) * 1.1/2)
    # R1 = Close + ((High-Low) * 1.0833/2)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High-Low) * 1.0833/2)
    # S2 = Close - ((High-Low) * 1.1/2)
    # S3 = Close - ((High-Low) * 1.25/2)
    # S4 = Close - ((High-Low) * 1.5/2)
    
    # We only need R1, S1, and PP for this strategy
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.0833 / 2.0)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.0833 / 2.0)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price at or below S1, 1d EMA34 rising, volume filter
            long_cond = (close[i] <= camarilla_s1_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price at or above R1, 1d EMA34 falling, volume filter
            short_cond = (close[i] >= camarilla_r1_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back above PP (mean reversion complete)
            if close[i] > camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back below PP (mean reversion complete)
            if close[i] < camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals