#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# Uses 1d Donchian(20) for breakout signals, 1w EMA(21) for trend direction
# Volume confirmation requires 1d volume > 1.5x 20-period average
# ATR filter ensures sufficient volatility (ATR > 20-period average)
# Fixed position size of 0.25 to balance return and drawdown
# Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years)

name = "1d_1w_donchian_trend_volume_v1"
timeframe = "1d"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 21:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    highest_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA (21-period) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 1d timeframe
    highest_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_1d)
    lowest_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_1d)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_1d_aligned[i]) or np.isnan(lowest_1d_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            atr_14_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 20-period average
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean()
        if len(atr_ma_20) > i:
            vol_filter = atr_14_1d_aligned[i] > atr_ma_20.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to midpoint of 1d channel
            midpoint_1d = (highest_1d_aligned[i] + lowest_1d_aligned[i]) / 2.0
            if close[i] < midpoint_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midpoint of 1d channel
            midpoint_1d = (highest_1d_aligned[i] + lowest_1d_aligned[i]) / 2.0
            if close[i] > midpoint_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume, volatility and trend confirmation
            if volume_confirmed:
                # Breakout above upper channel with uptrend (buy)
                if close[i] > highest_1d_aligned[i] and close_1d[i] > ema_21_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below lower channel with downtrend (sell)
                elif close[i] < lowest_1d_aligned[i] and close_1d[i] < ema_21_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals