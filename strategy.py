#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Williams %R overbought/oversold + volume spike + low volatility filter
# Targets: Fewer trades (20-50/year) to avoid fee drag, works in bull/bear via volatility regime filter
# Entry: Williams %R crosses below -80 (oversold) with 1.5x average volume in low volatility (ATR ratio < 0.6)
# Exit: Williams %R crosses above -20 (overbought) for longs, crosses below -80 for shorts
# Position size: 0.25 (25%) to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Williams %R (14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # Daily ATR (14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily ATR ratio (ATR / 50-period MA of ATR) for volatility regime
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / np.where(atr_ma_50_1d > 0, atr_ma_50_1d, np.nan)
    
    # Daily volume moving average (20) for volume spike filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Get aligned daily indicators
        williams_r_i = align_htf_to_ltf(prices, df_1d, williams_r)[i]
        atr_ratio_1d_i = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)[i]
        vol_ma_20_1d_i = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)[i]
        
        if np.isnan(williams_r_i) or np.isnan(atr_ratio_1d_i) or np.isnan(vol_ma_20_1d_i):
            continue
        
        # Volatility filter: only trade when ATR ratio < 0.6 (low volatility regime)
        low_vol = atr_ratio_1d_i < 0.6
        
        # Volume spike filter (1.5x daily average volume)
        volume_spike = volume[i] > 1.5 * vol_ma_20_1d_i
        
        # Long: Williams %R crosses below -80 (oversold) with volume spike in low vol
        if position == 0 and low_vol and volume_spike:
            if williams_r_i < -80:
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses above -20 (overbought) with volume spike in low vol
            elif williams_r_i > -20:
                position = -1
                signals[i] = -position_size
        
        # Exit: Williams %R crosses above -20 for longs, below -80 for shorts
        elif position != 0:
            if position == 1 and williams_r_i > -20:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r_i < -80:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_Oversold_Overbought_Volume_Spike_LowVol"
timeframe = "12h"
leverage = 1.0