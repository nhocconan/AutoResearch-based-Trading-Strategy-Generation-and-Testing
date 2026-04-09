#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d/1w Supertrend + volume confirmation + ATR filter
# Supertrend captures trend direction with built-in ATR-based stop/reversal
# Weekly Supertrend sets major trend bias, daily confirms intermediate trend
# Volume confirmation (current 6h volume > 1.4x 20-period average) filters weak breakouts
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Position size fixed at 0.25 to balance return and drawdown
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_1d_supertrend_volume_atr_v1"
timeframe = "6h"
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
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend for 1w (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # 1w ATR
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1w = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # 1w Supertrend
    hl2_1w = (high_1w + low_1w) / 2.0
    upper_band_1w = hl2_1w + (multiplier * atr_1w)
    lower_band_1w = hl2_1w - (multiplier * atr_1w)
    
    supertrend_1w = np.zeros_like(close_1w)
    direction_1w = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        # Upper band
        if close_1w[i-1] <= upper_band_1w[i-1]:
            upper_band_1w[i] = min(upper_band_1w[i], upper_band_1w[i-1])
        else:
            upper_band_1w[i] = upper_band_1w[i]
            
        # Lower band
        if close_1w[i-1] >= lower_band_1w[i-1]:
            lower_band_1w[i] = max(lower_band_1w[i], lower_band_1w[i-1])
        else:
            lower_band_1w[i] = lower_band_1w[i]
            
        # Trend
        if close_1w[i] <= lower_band_1w[i-1]:
            direction_1w[i] = -1
        elif close_1w[i] >= upper_band_1w[i-1]:
            direction_1w[i] = 1
        else:
            direction_1w[i] = direction_1w[i-1]
            
        if direction_1w[i] == 1:
            supertrend_1w[i] = lower_band_1w[i]
        else:
            supertrend_1w[i] = upper_band_1w[i]
    
    # Calculate Supertrend for 1d (ATR=10, multiplier=3.0)
    # 1d ATR
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # 1d Supertrend
    hl2_1d = (high_1d + low_1d) / 2.0
    upper_band_1d = hl2_1d + (multiplier * atr_1d)
    lower_band_1d = hl2_1d - (multiplier * atr_1d)
    
    supertrend_1d = np.zeros_like(close_1d)
    direction_1d = np.ones_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Upper band
        if close_1d[i-1] <= upper_band_1d[i-1]:
            upper_band_1d[i] = min(upper_band_1d[i], upper_band_1d[i-1])
        else:
            upper_band_1d[i] = upper_band_1d[i]
            
        # Lower band
        if close_1d[i-1] >= lower_band_1d[i-1]:
            lower_band_1d[i] = max(lower_band_1d[i], lower_band_1d[i-1])
        else:
            lower_band_1d[i] = lower_band_1d[i]
            
        # Trend
        if close_1d[i] <= lower_band_1d[i-1]:
            direction_1d[i] = -1
        elif close_1d[i] >= upper_band_1d[i-1]:
            direction_1d[i] = 1
        else:
            direction_1d[i] = direction_1d[i-1]
            
        if direction_1d[i] == 1:
            supertrend_1d[i] = lower_band_1d[i]
        else:
            supertrend_1d[i] = upper_band_1d[i]
    
    # Calculate 1w ATR (14-period) for volatility filtering
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_14_1w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_14_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(supertrend_1w_aligned[i]) or np.isnan(direction_1w_aligned[i]) or
            np.isnan(supertrend_1d_aligned[i]) or np.isnan(direction_1d_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_1w_aligned[i] <= 0 or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.4x average 6h volume
        volume_confirmed = volume[i] > 1.4 * vol_ma_20[i]
        
        # Volatility filter: only trade when both 1w and 1d ATR are above their 30-period averages
        atr_ma_30_1w = pd.Series(atr_1w_aligned).rolling(window=30, min_periods=30).mean()
        atr_ma_30_1d = pd.Series(atr_1d_aligned).rolling(window=30, min_periods=30).mean()
        if len(atr_ma_30_1w) > i and len(atr_ma_30_1d) > i:
            vol_filter = (atr_1w_aligned[i] > atr_ma_30_1w.iloc[i]) and (atr_1d_aligned[i] > atr_ma_30_1d.iloc[i])
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on trend reversal (both 1w and 1d turn down)
            if direction_1w_aligned[i] == -1 and direction_1d_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on trend reversal (both 1w and 1d turn up)
            if direction_1w_aligned[i] == 1 and direction_1d_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Enter when both timeframes agree on trend direction with volume confirmation
            if volume_confirmed:
                if direction_1w_aligned[i] == 1 and direction_1d_aligned[i] == 1:
                    position = 1
                    signals[i] = position_size
                elif direction_1w_aligned[i] == -1 and direction_1d_aligned[i] == -1:
                    position = -1
                    signals[i] = -position_size
    
    return signals