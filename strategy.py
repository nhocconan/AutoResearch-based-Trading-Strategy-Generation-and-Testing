#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h/1d Donchian breakout with volume confirmation and ATR filter
# Donchian(20) breakout captures momentum, volume > 1.5x 20-period average confirms strength
# ATR filter ensures sufficient volatility (> ATR MA50) to avoid choppy periods
# Position size fixed at 0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_12h_1d_donchian_breakout_volume_atr_v1"
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
    
    # Calculate 12h Donchian channels (20-period)
    highest_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Donchian channels (20-period)
    highest_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ATR (14-period) for volatility filtering
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    highest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_20_12h)
    lowest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_20_12h)
    highest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_20_1d)
    lowest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20_1d)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(highest_20_12h_aligned[i]) or np.isnan(lowest_20_12h_aligned[i]) or
            np.isnan(highest_20_1d_aligned[i]) or np.isnan(lowest_20_1d_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_12h_aligned[i] <= 0 or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when both 12h and 1d ATR are above their 50-period averages
        atr_ma_50_12h = pd.Series(atr_12h_aligned).rolling(window=50, min_periods=50).mean()
        atr_ma_50_1d = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50_12h) > i and len(atr_ma_50_1d) > i:
            vol_filter = (atr_12h_aligned[i] > atr_ma_50_12h.iloc[i]) and (atr_1d_aligned[i] > atr_ma_50_1d.iloc[i])
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to 12h or 1d lowest Donchian level
            if close[i] < lowest_20_12h_aligned[i] or close[i] < lowest_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to 12h or 1d highest Donchian level
            if close[i] > highest_20_12h_aligned[i] or close[i] > highest_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above highest Donchian level (long)
                if close[i] > highest_20_12h_aligned[i] or close[i] > highest_20_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below lowest Donchian level (short)
                elif close[i] < lowest_20_12h_aligned[i] or close[i] < lowest_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals