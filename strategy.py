#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d Volume Spike + ADX Regime.
Long when Williams %R < -80 (oversold) + 1d volume > 1.5x 20-period average + 1d ADX > 25 (trend) + price > 1d EMA50.
Short when Williams %R > -20 (overbought) + 1d volume > 1.5x 20-period average + 1d ADX > 25 (trend) + price < 1d EMA50.
In range regimes (1d ADX < 20), fade extremes: long at %R < -90, short at %R > -10.
Uses 1d for volume/ADX/EMA filters, 6h for Williams %R timing.
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
    
    # Get 1d data for regime filters (volume, ADX, EMA)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
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
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.divide(volume_1d, vol_ma_20, out=np.ones_like(volume_1d), where=vol_ma_20!=0)
    
    # Align 1d indicators
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 6h Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_14 = calculate_williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(wr_14[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        adx_val = adx_14_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        wr_val = wr_14[i]
        price = close[i]
        
        # Trend regime: ADX > 25
        # Range regime: ADX < 20
        is_trend = adx_val > 25
        is_range = adx_val < 20
        
        if position == 0:
            # Long conditions
            if is_trend:
                # Trend long: oversold + volume spike + price > EMA50
                if wr_val < -80 and vol_ratio > 1.5 and price > ema50_val:
                    signals[i] = 0.25
                    position = 1
            elif is_range:
                # Range long: deep oversold
                if wr_val < -90:
                    signals[i] = 0.25
                    position = 1
            
            # Short conditions
            if is_trend:
                # Trend short: overbought + volume spike + price < EMA50
                if wr_val > -20 and vol_ratio > 1.5 and price < ema50_val:
                    signals[i] = -0.25
                    position = -1
            elif is_range:
                # Range short: deep overbought
                if wr_val > -10:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral range OR regime shifts against position
            if wr_val > -50 or (is_trend and price < ema50_val) or (is_range and wr_val > -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral range OR regime shifts against position
            if wr_val < -50 or (is_trend and price > ema50_val) or (is_range and wr_val < -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dVolumeSpike_ADXRegime"
timeframe = "6h"
leverage = 1.0