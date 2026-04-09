#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and ATR filter
# Donchian(20) on 1d provides major support/resistance levels that work in both bull and bear markets
# Breakout above 20-day high = long, breakdown below 20-day low = short
# Volume confirmation (current 12h volume > 1.4x 20-period average) filters false breakouts
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Position size fixed at 0.25 to balance risk and reward
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_volume_atr_v2"
timeframe = "12h"
leverage = 1.0

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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = 20-day high
    # Lower band = 20-day low
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian levels and ATR to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.4x average 12h volume
        volume_confirmed = volume[i] > 1.4 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low-vol chop)
        atr_ma_50 = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50) > i:
            vol_filter = atr_aligned[i] > atr_ma_50.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size (discrete level to minimize fee churn)
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to lower band or stop at lower band breakdown
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to upper band or stop at upper band breakout
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout trading with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above upper band (buy breakout)
                if close[i] > upper_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakdown below lower band (sell breakdown)
                elif close[i] < lower_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals