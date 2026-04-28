#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Adaptive
Hypothesis: 4h breakouts at R3/S3 levels with 1d trend filter and volume confirmation, using adaptive position sizing based on volatility regime to balance performance in bull and bear markets. Targets 20-40 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price action and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate ATR for volatility regime and position sizing
    atr_period = 14
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Camarilla levels (using previous day's range)
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    R3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4
    S3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # Align all higher timeframe data to 4h
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    d_uptrend = close > ema_50_1d_aligned
    d_downtrend = close < ema_50_1d_aligned
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    # Volatility regime filter: only trade when ATR is above its 50-period median
    atr_median_50 = pd.Series(atr_aligned).rolling(window=50, min_periods=50).median().values
    vol_regime = atr_aligned > atr_median_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i]) or
            np.isnan(vol_regime[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment, volume surge, and volatility regime
        # Long: price breaks above R3 + daily uptrend + volume surge + volatility regime
        long_entry = (close[i] > R3_aligned[i] and 
                     d_uptrend[i] and 
                     volume_surge[i] and
                     vol_regime[i])
        
        # Short: price breaks below S3 + daily downtrend + volume surge + volatility regime
        short_entry = (close[i] < S3_aligned[i] and 
                      d_downtrend[i] and 
                      volume_surge[i] and
                      vol_regime[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
        # Adaptive position sizing based on volatility (inverse volatility scaling)
        base_size = 0.25
        vol_scalar = np.clip(atr_median_50[i] / (atr_aligned[i] + 1e-10), 0.5, 2.0)
        position_size = base_size * vol_scalar
        position_size = np.clip(position_size, 0.15, 0.35)  # Keep within reasonable bounds
        
        if long_entry and position <= 0:
            signals[i] = position_size
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -position_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = -position_size  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = position_size   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Adaptive"
timeframe = "4h"
leverage = 1.0