#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal + 1d Volume Spike + ADX Trend Filter
Hypothesis: Camarilla levels from daily timeframe provide high-probability
reversal zones. Combined with volume confirmation and ADX trend strength,
this captures mean reversion in ranging markets and avoids false signals
in strong trends. Designed for low frequency (~20-30/year) to minimize
fee drag and works in both bull (buy dips in uptrend) and bear (sell rallies
in downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smoothing
        tr_period = np.full(n, np.nan)
        dm_plus_period = np.full(n, np.nan)
        dm_minus_period = np.full(n, np.nan)
        
        if n >= period + 1:
            tr_period[period] = np.sum(tr[:period])
            dm_plus_period[period] = np.sum(dm_plus[:period])
            dm_minus_period[period] = np.sum(dm_minus[:period])
            
            for i in range(period + 1, n):
                tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i-1]
                dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i-1]
                dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i-1]
        
        # Directional Indicators
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        if n >= period + 1:
            di_plus[period:] = (dm_plus_period[period:] / tr_period[period:]) * 100
            di_minus[period:] = (dm_minus_period[period:] / tr_period[period:]) * 100
            dx[period:] = np.where(
                (di_plus[period:] + di_minus[period:]) > 0,
                np.abs(di_plus[period:] - di_minus[period:]) / (di_plus[period:] + di_minus[period:]) * 100,
                0
            )
        
        # ADX
        adx = np.full(n, np.nan)
        if n >= 2 * period + 1:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # 20-period volume average for volume filter
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    camarilla_S1 = np.full(len(close_1d), np.nan)
    camarilla_S2 = np.full(len(close_1d), np.nan)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    camarilla_S4 = np.full(len(close_1d), np.nan)
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_R2 = np.full(len(close_1d), np.nan)
    camarilla_R3 = np.full(len(close_1d), np.nan)
    camarilla_R4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i > 0 and not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            range_ = high_1d[i] - low_1d[i]
            close_prev = close_1d[i-1]
            
            camarilla_S1[i] = close_prev - (range_ * 1.0833 / 2)
            camarilla_S2[i] = close_prev - (range_ * 1.1666 / 2)
            camarilla_S3[i] = close_prev - (range_ * 1.2500 / 2)
            camarilla_S4[i] = close_prev - (range_ * 1.5000 / 2)
            camarilla_R1[i] = close_prev + (range_ * 1.0833 / 2)
            camarilla_R2[i] = close_prev + (range_ * 1.1666 / 2)
            camarilla_R3[i] = close_prev + (range_ * 1.2500 / 2)
            camarilla_R4[i] = close_prev + (range_ * 1.5000 / 2)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_S1_12h = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_S2_12h = align_htf_to_ltf(prices, df_1d, camarilla_S2)
    camarilla_S3_12h = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_S4_12h = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_R1_12h = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_R2_12h = align_htf_to_ltf(prices, df_1d, camarilla_R2)
    camarilla_R3_12h = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_R4_12h = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(34, 20)  # ADX needs ~2*period, volume MA needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(camarilla_S1_12h[i]) or np.isnan(camarilla_R1_12h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (current volume > 2x 20-period average)
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # ADX filter: only trade when trend is weak (ADX < 25) for mean reversion
        adx_filter = adx[i] < 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 or S4 level OR ADX strengthens (trend resumes)
            # Stoploss: price drops 2*ATR below entry (using 14-period ATR approx)
            atr_approx = np.abs(high[i] - low[i])  # Simple range approximation
            if (close[i] <= camarilla_S3_12h[i] or
                close[i] <= camarilla_S4_12h[i] or
                adx[i] >= 30 or
                close[i] < entry_price - 2.0 * atr_approx):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price reaches R3 or R4 level OR ADX strengthens (trend resumes)
            # Stoploss: price rises 2*ATR above entry
            atr_approx = np.abs(high[i] - low[i])  # Simple range approximation
            if (close[i] >= camarilla_R3_12h[i] or
                close[i] >= camarilla_R4_12h[i] or
                adx[i] >= 30 or
                close[i] > entry_price + 2.0 * atr_approx):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Camarilla level touch + volume spike + weak trend (ADX < 25)
            # Minimum holding period: only allow new entry after 24 bars flat (2 days)
            if bars_since_entry >= 24:
                # Long: price touches S1 or S2 with volume spike in weak trend
                if ((abs(close[i] - camarilla_S1_12h[i]) < 0.001 * close[i] or
                     abs(close[i] - camarilla_S2_12h[i]) < 0.001 * close[i]) and
                    volume_filter and adx_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: price touches R1 or R2 with volume spike in weak trend
                elif ((abs(close[i] - camarilla_R1_12h[i]) < 0.001 * close[i] or
                       abs(close[i] - camarilla_R2_12h[i]) < 0.001 * close[i]) and
                      volume_filter and adx_filter):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals