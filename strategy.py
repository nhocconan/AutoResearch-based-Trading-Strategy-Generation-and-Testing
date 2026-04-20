#!/usr/bin/env python3
# 12h_1w_1d_VWAP_Reversion_Trend
# Hypothesis: Price reverts to weekly VWAP during ranging markets (identified by weekly ADX < 20) with 1d volume confirmation.
# In trending markets (weekly ADX > 25), follow the trend by buying when price pulls back to weekly VWAP from above.
# Works in both bull and bear: mean reversion in ranges, trend following on pullbacks.
# Uses 12h timeframe for execution, targeting 15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_VWAP_Reversion_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly VWAP (Volume Weighted Average Price) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Typical price for VWAP calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_numerator = np.cumsum(typical_price_1w * volume_1w)
    vwap_denominator = np.cumsum(volume_1w)
    vwap_1w = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, np.nan)
    
    # === Weekly ADX for trend strength ===
    # Calculate +DM, -DM, TR
    high_shift = np.roll(high_1w, 1)
    low_shift = np.roll(low_1w, 1)
    high_shift[0] = high_1w[0]
    low_shift[0] = low_1w[0]
    
    plus_dm = np.where((high_1w - high_shift) > (low_shift - low_1w), np.maximum(high_1w - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low_1w) > (high_1w - high_shift), np.maximum(low_shift - low_1w, 0), 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (equivalent to RMA)
    def rma(series, period):
        result = np.full_like(series, np.nan)
        if len(series) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(series[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    period_adx = 14
    atr_1w = rma(tr, period_adx)
    plus_di_1w = 100 * rma(plus_dm, period_adx) / np.where(atr_1w != 0, atr_1w, np.nan)
    minus_di_1w = 100 * rma(minus_dm, period_adx) / np.where(atr_1w != 0, atr_1w, np.nan)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / np.where((plus_di_1w + minus_di_1w) != 0, (plus_di_1w + minus_di_1w), np.nan)
    adx_1w = rma(dx_1w, period_adx)
    
    # === Daily Volume Ratio (current vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma20_1d > 0, vol_ma20_1d, np.nan)
    
    # Align all weekly and daily data to 12h
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        vwap_val = vwap_1w_aligned[i]
        adx_val = adx_1w_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vwap_val) or np.isnan(adx_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. In ranging market (ADX < 20): price near VWAP with volume confirmation (mean reversion)
            # 2. In trending market (ADX > 25): price pulls back to VWAP from above with volume (trend following)
            price_vwap_ratio = close_val / vwap_val
            
            if (adx_val < 20 and  # Ranging market
                0.98 <= price_vwap_ratio <= 1.02 and  # Near VWAP (±2%)
                vol_ratio_val > 1.3):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            elif (adx_val > 25 and  # Trending market
                  price_vwap_ratio > 1.0 and  # Price above VWAP
                  1.0 <= price_vwap_ratio <= 1.015 and  # Pullback to VWAP from above (0-1.5%)
                  vol_ratio_val > 1.3):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short conditions (symmetric)
            elif (adx_val < 20 and  # Ranging market
                  0.98 <= price_vwap_ratio <= 1.02 and  # Near VWAP (±2%)
                  vol_ratio_val > 1.3):  # Volume confirmation
                signals[i] = -0.25
                position = -1
            elif (adx_val > 25 and  # Trending market
                  price_vwap_ratio < 1.0 and  # Price below VWAP
                  0.985 <= price_vwap_ratio <= 1.0 and  # Pullback to VWAP from below (-1.5% to 0%)
                  vol_ratio_val > 1.3):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price moves significantly away from VWAP or ADX weakens
            price_vwap_ratio = close_val / vwap_val
            if price_vwap_ratio > 1.03 or price_vwap_ratio < 0.99 or adx_val < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price moves significantly away from VWAP or ADX weakens
            price_vwap_ratio = close_val / vwap_val
            if price_vwap_ratio < 0.97 or price_vwap_ratio > 1.01 or adx_val < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals