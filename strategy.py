#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes with volume confirmation and ATR-based stops
# Williams %R identifies overbought/oversold conditions that work in ranging markets
# Volume confirmation ensures breakouts have conviction
# ATR stoploss manages risk in both bull and bear markets
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Handle division by zero
    
    # Calculate 1d ATR (14-period) for volatility filtering and position sizing
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Williams %R and ATR to 4h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(wr_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma_20[i]) or atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x average 4h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 30-period average (avoid low-vol chop)
        atr_ma_30 = pd.Series(atr_aligned).rolling(window=30, min_periods=30).mean()
        if len(atr_ma_30) > i:
            vol_filter = atr_aligned[i] > atr_ma_30.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size: 0.25 (25% of capital)
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on Williams %R crossing above -20 (overbought) or ATR stoploss
            if wr_aligned[i] > -20 or close[i] < prices['low'].iloc[i-1] - 1.5 * atr_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on Williams %R crossing below -80 (oversold) or ATR stoploss
            if wr_aligned[i] < -80 or close[i] > prices['high'].iloc[i-1] + 1.5 * atr_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion trading with volume confirmation
            # Long when Williams %R < -80 (oversold), Short when Williams %R > -20 (overbought)
            if volume_confirmed:
                if wr_aligned[i] < -80:
                    position = 1
                    signals[i] = position_size
                elif wr_aligned[i] > -20:
                    position = -1
                    signals[i] = -position_size
    
    return signals