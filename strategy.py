#!/usr/bin/env python3
# 12h_Camarilla_MeanReversion_V1
# Hypothesis: In low-volatility regimes (Choppiness Index > 61.8), price mean-reverts at daily Camarilla pivot levels (R1/S1).
# Entries occur when price touches R1/S1 with rejection (close inside prior bar's range) and volume confirmation.
# Exits on opposite touch or volatility expansion (Choppiness Index < 38.2). Designed for low trade frequency (<20/year) to minimize fee drag.
# Works in both bull/bear by focusing on mean-reversion in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Choppiness Index for regime filter
    def calculate_choppiness(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR (smoothed TR)
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of true ranges over period
        tr_sum = np.full_like(tr, np.nan)
        if len(tr) >= period:
            tr_sum[period-1] = np.nansum(tr[1:period+1])
            for i in range(period, len(tr)):
                tr_sum[i] = tr_sum[i-1] + tr[i] - tr[i-period]
        
        # Max/min close over period
        max_close = np.full_like(close, np.nan)
        min_close = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i >= period-1:
                max_close[i] = np.max(close[i-period+1:i+1])
                min_close[i] = np.min(close[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.full_like(tr, np.nan)
        valid = ~np.isnan(atr) & (atr != 0) & ~np.isnan(max_close) & ~np.isnan(min_close) & ((max_close - min_close) != 0)
        chop[valid] = 100 * np.log10(tr_sum[valid] / (atr[valid] * period)) / np.log10(period)
        return chop
    
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Calculate Camarilla levels (based on previous day)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            R1[i] = prev_close + 1.1 * range_ / 12
            S1[i] = prev_close - 1.1 * range_ / 12
    
    # Align all 1d data to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 14) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(chop_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: choppy market (Choppiness Index > 61.8)
        choppy = chop_12h[i] > 61.8
        
        if position == 0:
            # Long: price touches or goes below S1 but closes inside prior bar's range (rejection)
            if low[i] <= S1_12h[i] and close[i] > low[i] and vol_confirm and choppy:
                # Additional confirmation: close inside prior bar's range (not breaking through)
                if i > 0 and close[i] >= low[i-1] and close[i] <= high[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Short: price touches or goes above R1 but closes inside prior bar's range (rejection)
            elif high[i] >= R1_12h[i] and close[i] < high[i] and vol_confirm and choppy:
                # Additional confirmation: close inside prior bar's range (not breaking through)
                if i > 0 and close[i] >= low[i-1] and close[i] <= high[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price touches or goes above R1 OR market becomes trending (chop < 38.2)
            if high[i] >= R1_12h[i] or chop_12h[i] < 38.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or goes below S1 OR market becomes trending (chop < 38.2)
            if low[i] <= S1_12h[i] or chop_12h[i] < 38.2:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_MeanReversion_V1"
timeframe = "12h"
leverage = 1.0