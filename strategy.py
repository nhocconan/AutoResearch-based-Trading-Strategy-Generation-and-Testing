#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R2S2_Breakout_Volume_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Calculate Previous Day's Pivot Points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Align 1d indicators to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 6h: ATR for volatility filter ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 6h: ADX for trend strength filter ===
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / np.where(tr_14 > 0, tr_14, np.nan)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / np.where(tr_14 > 0, tr_14, np.nan)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) > 0, (plus_di_14 + minus_di_14), np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        pivot_level = pivot_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        adx_val = adx[i]
        
        # Skip if any value is NaN
        if (np.isnan(r2_level) or np.isnan(s2_level) or np.isnan(pivot_level) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median (avoid choppy markets)
        atr_median = np.nanmedian(atr[max(0, i-49):i+1])
        vol_filter = atr_val > atr_median if not np.isnan(atr_median) else False
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: Price breaks above R2 with volume confirmation, volatility filter, and trend filter
            if (close_val > r2_level and   # Break above R2
                vol_ratio_val > 1.5 and    # Volume confirmation
                vol_filter and             # Volatility filter
                trend_filter):             # Trend filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 with volume confirmation, volatility filter, and trend filter
            elif (close_val < s2_level and   # Break below S2
                  vol_ratio_val > 1.5 and    # Volume confirmation
                  vol_filter and             # Volatility filter
                  trend_filter):             # Trend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below pivot (mean reversion to pivot)
            if close_val < pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above pivot (mean reversion to pivot)
            if close_val > pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals