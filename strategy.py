#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and ATR filter
# Weekly Donchian channels (20-period) act as major trend filters in both bull and bear markets
# Breakouts above/below weekly Donchian bands with volume confirmation signal strong moves
# ATR filter ensures sufficient volatility, avoiding false signals in low-vol periods
# Position size fixed at 0.28 to balance return and drawdown
# Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

name = "1d_1w_donchian_breakout_volume_atr_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_ma_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_ma_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR (14-period) for volatility filtering
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 1d timeframe
    upper_20_1w_aligned = align_htf_to_ltf(prices, df_1w, high_ma_20_1w)
    lower_20_1w_aligned = align_htf_to_ltf(prices, df_1w, low_ma_20_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR filter (50-period average for volatility regime)
    atr_ma_50 = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_1w_aligned[i]) or np.isnan(lower_20_1w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_ma_50[i]) or atr_1w_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when current ATR is above its 50-period average
        vol_filter = atr_1w_aligned[i] > atr_ma_50[i]
        
        # Fixed position size to minimize fee churn
        position_size = 0.28
        
        if position == 1:  # Long position
            # Exit on retracement to weekly lower Donchian band
            if close[i] < lower_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to weekly upper Donchian band
            if close[i] > upper_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Weekly Donchian breakout with volume and volatility confirmation
            if volume_confirmed and vol_filter:
                # Breakout above weekly upper band (long)
                if close[i] > upper_20_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below weekly lower band (short)
                elif close[i] < lower_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals