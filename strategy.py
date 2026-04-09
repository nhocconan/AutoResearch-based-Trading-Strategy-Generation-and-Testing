#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes + volume confirmation + ATR filter
# Williams %R > -20 = overbought (fade short), < -80 = oversold (fade long)
# Volume confirmation: current 4h volume > 1.5x 20-period average filters low-vol noise
# ATR filter: only trade when 1d ATR(14) > its 50-period average ensures sufficient volatility
# Fixed position size 0.25 to minimize fee churn and manage drawdown
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_williamsr_volume_atr_v1"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = highest_high_1d - lowest_low_1d
    williams_r_1d = np.where(denominator != 0, -100 * (highest_high_1d - close_1d) / denominator, -50)
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF data to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR 50-period average for volatility filter
    atr_ma_50_1d = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_ma_50_1d[i]) or
            not in_session[i] or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when 1d ATR > its 50-period average
        vol_filter = atr_1d_aligned[i] > atr_ma_50_1d[i]
        
        if not (volume_confirmed and vol_filter):
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to -50 level (mean reversion) or stop at -80 breakdown
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            elif williams_r_aligned[i] < -80:  # Stop loss at extreme oversold
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to -50 level (mean reversion) or stop at -20 breakout
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            elif williams_r_aligned[i] > -20:  # Stop loss at extreme overbought
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Williams %R fading strategy with volume and volatility confirmation
            # Fade at > -20 (sell at overbought, expect reversion to -50)
            # Fade at < -80 (buy at oversold, expect reversion to -50)
            if volume_confirmed and vol_filter:
                if williams_r_aligned[i] > -20:  # Overbought - fade short
                    position = -1
                    signals[i] = -position_size
                elif williams_r_aligned[i] < -80:  # Oversold - fade long
                    position = 1
                    signals[i] = position_size
    
    return signals