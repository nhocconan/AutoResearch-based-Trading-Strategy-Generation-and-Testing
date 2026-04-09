#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h and 1d Donchian channel breakouts with volume confirmation and ATR-based stoploss
# Donchian(20) breakouts capture strong momentum moves in both bull and bear markets
# Volume confirmation (current 4h volume > 1.5x 20-period average) filters false breakouts
# ATR stoploss (2.5x ATR) manages risk during volatile periods
# Uses 12h trend filter: only take long when price > 12h Donchian middle, short when price < middle
# Position size fixed at 0.25 to balance reward/risk and minimize fee churn
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_12h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channel (20-period)
    highest_12h_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_12h_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_12h_20 = (highest_12h_20 + lowest_12h_20) / 2.0
    
    # Calculate 1d Donchian channel (20-period)
    highest_1d_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_1d_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_1d_20 = (highest_1d_20 + lowest_1d_20) / 2.0
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    highest_12h_20_aligned = align_htf_to_ltf(prices, df_12h, highest_12h_20)
    lowest_12h_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_12h_20)
    middle_12h_20_aligned = align_htf_to_ltf(prices, df_12h, middle_12h_20)
    
    highest_1d_20_aligned = align_htf_to_ltf(prices, df_1d, highest_1d_20)
    lowest_1d_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_1d_20)
    middle_1d_20_aligned = align_htf_to_ltf(prices, df_1d, middle_1d_20)
    
    atr_14_aligned = align_htf_to_ltf(prices, prices, atr_14)  # Already 4h, but align for consistency
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_12h_20_aligned[i]) or np.isnan(lowest_12h_20_aligned[i]) or
            np.isnan(middle_12h_20_aligned[i]) or
            np.isnan(highest_1d_20_aligned[i]) or np.isnan(lowest_1d_20_aligned[i]) or
            np.isnan(middle_1d_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20[i]) or
            atr_14_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to 12h middle or stoploss hit
            if close[i] < middle_12h_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < low[i] - 2.5 * atr_14_aligned[i]:  # ATR-based stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to 12h middle or stoploss hit
            if close[i] > middle_12h_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > high[i] + 2.5 * atr_14_aligned[i]:  # ATR-based stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with 12h trend filter and volume confirmation
            # Long breakout: price > 1d upper AND price > 12h middle (uptrend filter)
            if (close[i] > highest_1d_20_aligned[i] and 
                close[i] > middle_12h_20_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short breakout: price < 1d lower AND price < 12h middle (downtrend filter)
            elif (close[i] < lowest_1d_20_aligned[i] and 
                  close[i] < middle_12h_20_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
    
    return signals