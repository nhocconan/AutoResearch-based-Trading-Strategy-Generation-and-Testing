#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ATR-based volatility filter
# Donchian(20) on 1d provides major support/resistance levels that work in both bull and bear markets
# Breakout above upper band = long, breakdown below lower band = short
# Volume confirmation (current 4h volume > 1.5x 20-period average) filters false breakouts
# ATR filter ensures sufficient volatility (current ATR > 50-period average ATR)
# Fixed position size of 0.25 to balance return and drawdown
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_breakout_volume_atr_v1"
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
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR filter (50-period average for 1d ATR)
    atr_ma_50_1d = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_ma_50_1d[i]) or
            not in_session[i] or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when current 1d ATR is above its 50-period average
        vol_filter = atr_1d_aligned[i] > atr_ma_50_1d[i]
        
        if not volume_confirmed or not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to midline or stop at lower band breakdown
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if close[i] < midline:
                position = 0
                signals[i] = 0.0
            elif close[i] < donchian_low_aligned[i]:  # Stop loss at lower band breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midline or stop at upper band breakout
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if close[i] > midline:
                position = 0
                signals[i] = 0.0
            elif close[i] > donchian_high_aligned[i]:  # Stop loss at upper band breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout trading with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above upper band (buy break above resistance)
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakdown below lower band (sell break below support)
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals