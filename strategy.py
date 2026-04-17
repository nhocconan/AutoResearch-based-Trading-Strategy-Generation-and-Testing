#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot R3/S3 Fade with 12h Volume Confirmation and ADX Regime Filter.
In ranging markets (ADX < 25), price tends to revert from extreme Camarilla levels (R3/S3).
In trending markets (ADX > 25), breakouts through R4/S4 with volume confirmation continue.
Uses 12h for Camarilla pivots (more stable than 1d) and 6h for entries.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Get 12h data for Camarilla pivots and ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ADX (10-period for slower TF)
    def calculate_adx(high, low, close, period=10):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        if len(tr) >= period+1:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Avoid division by zero
        atr_safe = np.where(atr == 0, 1e-10, atr)
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr_safe)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr_safe)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate Camarilla pivots for 12h
    def calculate_camarilla(high, low, close):
        # Typical price
        pp = (high + low + close) / 3.0
        range_ = high - low
        
        # Camarilla levels
        r4 = pp + range_ * 1.1 / 2
        r3 = pp + range_ * 1.1 / 4
        s3 = pp - range_ * 1.1 / 4
        s4 = pp - range_ * 1.1 / 2
        
        return r3, r4, s3, s4, pp
    
    # Calculate 12h indicators
    adx_10 = calculate_adx(high_12h, low_12h, close_12h, 10)
    r3_12h, r4_12h, s3_12h, s4_12h, pp_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Volume average (20-period)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h
    adx_10_aligned = align_htf_to_ltf(prices, df_12h, adx_10)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 6h volume ratio (current vs 20-period MA)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_6h > 0, volume / vol_ma_6h, 1.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_10_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_10_aligned[i]
        price = close[i]
        vol = vol_ratio[i]
        
        # Regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_val < 25
        is_trend = adx_val > 25
        
        # Volume confirmation: above average volume
        vol_confirmed = vol > 1.2
        
        if position == 0:
            # Long signals
            if is_range:
                # In range: fade from S3 (mean reversion long)
                if price <= s3_12h_aligned[i] and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
            else:  # is_trend
                # In trend: breakout above R4 with volume
                if price >= r4_12h_aligned[i] and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
            
            # Short signals
            if is_range:
                # In range: fade from R3 (mean reversion short)
                if price >= r3_12h_aligned[i] and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
            else:  # is_trend
                # In trend: breakdown below S4 with volume
                if price <= s4_12h_aligned[i] and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long conditions
            exit_signal = False
            if is_range:
                # Exit range long when price reaches midpoint or R3
                if price >= pp_12h_aligned[i] or price >= r3_12h_aligned[i]:
                    exit_signal = True
            else:
                # Exit trend long when price falls below R3 or loses volume
                if price < r3_12h_aligned[i] or not vol_confirmed:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short conditions
            exit_signal = False
            if is_range:
                # Exit range short when price reaches midpoint or S3
                if price <= pp_12h_aligned[i] or price <= s3_12h_aligned[i]:
                    exit_signal = True
            else:
                # Exit trend short when price rises above S3 or loses volume
                if price > s3_12h_aligned[i] or not vol_confirmed:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Volume_ADX_Regime"
timeframe = "6h"
leverage = 1.0