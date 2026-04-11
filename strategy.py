#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d ATR-based volatility filter + volume confirmation
# - Williams %R(14) identifies overbought/oversold conditions on 12h chart
# - Long: Williams %R crosses above -80 (oversold recovery) AND ATR(14) > 1.2 * ATR(50) (expanding volatility) AND volume > 1.3x 20-period average
# - Short: Williams %R crosses below -20 (overbought rejection) AND ATR(14) > 1.2 * ATR(50) (expanding volatility) AND volume > 1.3x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Williams %R captures mean reversion in ranging markets and momentum in trending markets
# - ATR ratio filter ensures we only trade during volatile expansion phases (avoids chop)
# - Volume confirmation filters out weak breakouts
# - Works in bull markets (buying dips) and bear markets (selling rallies)

name = "12h_williamsr_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ATR and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = 0  # First bar: no previous close
    tr3[0] = 0  # First bar: no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams %R on 12h timeframe
    highest_high = pd.Series(close).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(close).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R crossover signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # First bar: no previous value
    
    # Long signal: Williams %R crosses above -80 (from below)
    williams_r_cross_above_80 = (williams_r_prev <= -80) & (williams_r > -80)
    # Short signal: Williams %R crosses below -20 (from above)
    williams_r_cross_below_20 = (williams_r_prev >= -20) & (williams_r < -20)
    
    for i in range(60, n):  # Start after 60-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Volatility filter: ATR(14) > 1.2 * ATR(50) (expanding volatility)
        vol_filter = atr_ratio_aligned[i] > 1.2
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R crosses above -80 + volatility expansion + volume confirmation
        if williams_r_cross_above_80[i] and vol_filter and vol_confirm:
            enter_long = True
        
        # Short: Williams %R crosses below -20 + volatility expansion + volume confirmation
        if williams_r_cross_below_20[i] and vol_filter and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse Williams %R crossover or loss of volatility/volume
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R crosses below -80 OR volatility contracts OR volume weakens
            exit_long = (williams_r_cross_below_20[i]) or (not vol_filter) or (not vol_confirm)
        elif position == -1:
            # Exit short if Williams %R crosses above -20 OR volatility contracts OR volume weakens
            exit_short = (williams_r_cross_above_80[i]) or (not vol_filter) or (not vol_confirm)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals