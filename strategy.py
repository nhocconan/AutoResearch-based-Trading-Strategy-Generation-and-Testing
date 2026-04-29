#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1d ATR regime filter
# Long when price breaks above 20-period Donchian high AND volume > 1.5x 20-bar avg AND 1d ATR(14) < 0.8 * 20-bar ATR avg (low volatility breakout)
# Short when price breaks below 20-period Donchian low AND volume > 1.5x 20-bar avg AND 1d ATR(14) < 0.8 * 20-bar ATR avg
# Exit when price retests the midpoint of the Donchian channel
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Donchian breakouts capture strong momentum moves. Volume confirmation ensures breakout validity.
# Low volatility regime filter (1d ATR < 0.8 * 20-bar ATR avg) avoids choppy markets and false breakouts.
# Works in bull via upward breakouts, in bear via downward breakdowns. Novelty: combining Donchian with volatility regime on 12h timeframe.

name = "12h_Donchian20_VolumeConfirm_LowVolRegime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period ATR average for 12h timeframe (for regime threshold)
    # TR for 12h
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr1_12h[0] = high[0] - low[0]
    tr2_12h[0] = np.abs(high[0] - close[0])
    tr3_12h[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_20_12h = pd.Series(tr_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volatility regime: 1d ATR < 0.8 * 20-bar 12h ATR average (low volatility environment)
    vol_regime = atr_14_1d < 0.8 * atr_20_12h
    
    # Align 1d indicators to 12h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        vol_reg = vol_regime_aligned[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        curr_donch_mid = donchian_mid[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midpoint
            if curr_low <= curr_donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midpoint
            if curr_high >= curr_donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND volume confirmation AND low volatility regime
            if curr_high > curr_donch_high and vol_conf and vol_reg:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND volume confirmation AND low volatility regime
            elif curr_low < curr_donch_low and vol_conf and vol_reg:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals