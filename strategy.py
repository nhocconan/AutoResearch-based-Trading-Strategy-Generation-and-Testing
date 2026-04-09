#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ATR filter
# Donchian(20) on 1d acts as major support/resistance that works in both bull and bear markets
# Breakout above upper band = long, breakdown below lower band = short
# Volume confirmation (current 12h volume > 1.5x 20-period average) filters false signals
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Fixed position size 0.25 to balance return and drawdown
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_volume_atr_v3"
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
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when 1d ATR is above its 50-period average
        atr_ma_50_1d = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50_1d) > i:
            vol_filter = atr_1d_aligned[i] > atr_ma_50_1d.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to middle of channel or stop at lower band breakdown
            middle = (upper_20_aligned[i] + lower_20_aligned[i]) / 2.0
            if close[i] < middle:
                position = 0
                signals[i] = 0.0
            elif close[i] < lower_20_aligned[i]:  # Stop loss at lower band breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to middle of channel or stop at upper band breakout
            middle = (upper_20_aligned[i] + lower_20_aligned[i]) / 2.0
            if close[i] > middle:
                position = 0
                signals[i] = 0.0
            elif close[i] > upper_20_aligned[i]:  # Stop loss at upper band breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout trading with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above upper band (buy break above resistance)
                if close[i] > upper_20_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakdown below lower band (sell break below support)
                elif close[i] < lower_20_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals