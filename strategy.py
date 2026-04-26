#!/usr/bin/env python3
"""
6h_HTF_Regime_Camarilla_Breakout_v1
Hypothesis: 6h Camarilla H4/H5 breakout with 1d volatility regime filter and volume confirmation.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Regime filter: 1d ATR ratio (current ATR7 / ATR30) > 1.2 = high volatility (favor breakouts)
- Volume confirmation: 1d volume > 1.5x 20-period average
- Trend filter: price must be above/below 1d EMA50 for long/short respectively
- Designed to avoid low-volatility chop where breakouts fail, and high-fee overtrading
- Works in bull/bear markets by combining volatility expansion breakouts with trend alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d ATR for volatility regime (ATR7 and ATR30)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr7 / np.where(atr30 == 0, np.nan, atr30)  # Avoid division by zero
    
    # Volatility regime: high volatility when ATR7/ATR30 > 1.2
    vol_regime_high = atr_ratio > 1.2
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_high.astype(float))
    
    # Calculate 1d volume confirmation (>1.5x 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h4 = close_1d + (1.0/6) * (high_1d - low_1d)  # H4 = close + 1/6*(high-low)
    camarilla_h5 = close_1d + (1.0/4) * (high_1d - low_1d)  # H5 = close + 1/4*(high-low)
    camarilla_l4 = close_1d - (1.0/6) * (high_1d - low_1d)  # L4 = close - 1/6*(high-low)
    camarilla_l5 = close_1d - (1.0/4) * (high_1d - low_1d)  # L5 = close - 1/4*(high-low)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 30 for ATR30, 20 for volume MA)
    start_idx = max(50, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h5_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_l5_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_h5_aligned[i]
        breakout_short = close[i] < camarilla_l5_aligned[i]
        
        if position == 0:
            # Long: breakout above H5 AND close > 1d EMA50 AND high volatility regime AND volume spike
            if (breakout_long and 
                close[i] > ema50_1d_aligned[i] and 
                vol_regime_aligned[i] > 0.5 and 
                vol_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: breakout below L5 AND close < 1d EMA50 AND high volatility regime AND volume spike
            elif (breakout_short and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_regime_aligned[i] > 0.5 and 
                  vol_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below L4 OR volatility regime shifts to low
            if breakout_short or vol_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above H4 OR volatility regime shifts to low
            if breakout_long or vol_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_HTF_Regime_Camarilla_Breakout_v1"
timeframe = "6h"
leverage = 1.0