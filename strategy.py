#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_ADX
Hypothesis: 4h breakout at Camarilla R3/S3 levels (stronger levels) in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA) and ADX regime filter (ADX > 25 for trending markets). Uses discrete position sizing (0.25) to minimize fee drag. Designed for 15-30 trades/year to avoid fee drag while maintaining edge in both bull and bear regimes via strong breakout levels and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ADX for regime filter (14-period)
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr = np.maximum(np.abs(np.diff(high, prepend=high[0])), 
                    np.maximum(np.abs(np.diff(low, prepend=low[0])), 
                              np.abs(np.diff(close, prepend=close[0]))))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).rolling(window=14, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations
    start_idx = max(20, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_1d_aligned[i]
        adx_val = adx_14_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending_regime = adx_val > 25
        
        # Camarilla levels for R3 and S3 (based on previous day's range)
        if i >= 1:
            # Use previous bar's high, low, close for Camarilla calculation
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            # Camarilla R3 and S3 levels (stronger breakout levels)
            camarilla_r3 = prev_close + (range_val * 1.1 / 4)
            camarilla_s3 = prev_close - (range_val * 1.1 / 4)
        else:
            camarilla_r3 = high_val
            camarilla_s3 = low_val
        
        # Entry conditions: breakout at R3/S3 + trend + volume spike + trending regime
        long_entry = (close_val > camarilla_r3) and bullish_1d and vol_spike and trending_regime
        short_entry = (close_val < camarilla_s3) and bearish_1d and vol_spike and trending_regime
        
        # Exit conditions: opposite Camarilla level touch (R3 for shorts, S3 for longs)
        exit_long = close_val < camarilla_s3
        exit_short = close_val > camarilla_r3
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_ADX"
timeframe = "4h"
leverage = 1.0