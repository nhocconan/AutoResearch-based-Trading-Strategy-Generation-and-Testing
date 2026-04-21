#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 pivot breakout with 1d volume spike and 1w EMA trend filter.
# Captures institutional breakout at key daily pivot levels with volume confirmation.
# Works in both bull and bear markets by following higher timeframe trend.
# Target: 15-25 trades/year by requiring confluence of pivot breakout, volume surge, and EMA alignment.
# Entry: Long when price breaks above daily Camarilla R3 with volume spike and price > 1w EMA50.
#        Short when price breaks below daily Camarilla S3 with volume spike and price < 1w EMA50.
# Exit: Opposite pivot touch (S3 for long, R3 for short) or volume drops below average.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for pivot calculation and weekly data for EMA
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Pivot point
    pp = (high_d + low_d + close_d) / 3
    # Camarilla levels
    r3 = close_d + (high_d - low_d) * 1.1 / 2
    s3 = close_d - (high_d - low_d) * 1.1 / 2
    
    # Calculate 50-period EMA on weekly timeframe
    close_w = df_1w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation using daily volume
    vol_d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily and weekly data to 12h (wait for daily/weekly close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = align_htf_to_ltf(prices, df_1d, vol_d)[i]  # daily volume aligned to 12h
        
        # Trend filter: price relative to weekly EMA50
        above_ema = price_close > ema50_1w_aligned[i]
        below_ema = price_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = vol_current > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Enter long when price breaks above daily R3 with volume spike and above weekly EMA
            if (price_close > r3_aligned[i] and volume_confirm and above_ema):
                signals[i] = 0.25
                position = 1
            # Enter short when price breaks below daily S3 with volume spike and below weekly EMA
            elif (price_close < s3_aligned[i] and volume_confirm and below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches daily S3 (opposite side) or volume drops below average
                if price_close < s3_aligned[i]:
                    exit_signal = True
                elif vol_current < vol_ma_20_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price reaches daily R3 (opposite side) or volume drops below average
                if price_close > r3_aligned[i]:
                    exit_signal = True
                elif vol_current < vol_ma_20_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0