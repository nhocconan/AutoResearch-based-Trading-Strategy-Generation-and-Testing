#!/usr/bin/env python3
# 12h_camarilla_1w_trend_volume_v1
# Hypothesis: 12h strategy using 1w Camarilla pivot levels for structure, volume confirmation, and 1d trend filter (EMA50).
# Long: price above S3 + volume spike + close > 1d EMA50
# Short: price below S3 + volume spike + close < 1d EMA50
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for 1w (using previous week's OHLC)
    # Camarilla formula: 
    # H4 = close + 1.1*(high - low)/2
    # L4 = close - 1.1*(high - low)/2
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    # H2 = close + 1.1*(high - low)/6
    # L2 = close - 1.1*(high - low)/6
    # H1 = close + 1.1*(high - low)/12
    # L1 = close - 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12  (same as L1)
    # S2 = close - 1.1*(high - low)/6   (same as L2)
    # S3 = close - 1.1*(high - low)/4   (same as L3)
    # S4 = close - 1.1*(high - low)/2   (same as L4)
    
    # We need previous week's OHLC to calculate current week's levels
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Set first week's values to NaN (no previous week)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla S3 level (most important for reversals)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla S3 to 12h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 OR trend turns bearish
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 OR trend turns bullish
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price above S3 + bullish trend
                if close[i] > camarilla_s3_aligned[i] and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below S3 + bearish trend
                elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals