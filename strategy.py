#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1-day Volume Spike + 1-week Volatility Regime
# In high volatility (1w ATR ratio > 1.2): trade breakouts with volume confirmation
# In low volatility (1w ATR ratio < 0.8): avoid breakouts to prevent whipsaw
# Uses 4h price for Donchian channels, 1d volume for spike detection, 1w ATR for regime filter
# Designed to capture strong moves in volatile markets while avoiding choppy periods
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on daily timeframe
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_avg = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Load 1w data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for 1w ATR
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1w ATR ratio (current ATR / 50-period average ATR)
    atr_ma_50 = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1w = atr_1w / (atr_ma_50 + 1e-10)
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    # Calculate 4h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(vol_1d_avg[i]) or np.isnan(atr_ratio_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in high volatility (ATR ratio > 1.2)
        high_vol_regime = atr_ratio_1w_aligned[i] > 1.2
        
        price = close[i]
        
        if position == 0:
            # Enter long on Donchian breakout with volume confirmation
            long_signal = False
            if high_vol_regime:
                if price > donch_high[i] and vol_1d_avg[i] > 0:
                    # Current 1d volume > 20-day average volume
                    vol_spike = vol_1d_avg[i] > pd.Series(vol_1d).rolling(window=20, min_periods=1).mean().iloc[i] if i < len(pd.Series(vol_1d).rolling(window=20, min_periods=1).mean()) else False
                    # Simplified: use current volume > 1.5 * 20-day average as spike
                    if i >= 20:  # Ensure we have enough data for 20-day average
                        vol_ma_20_current = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().iloc[i]
                        if vol_1d_avg[i] > 1.5 * vol_ma_20_current:
                            long_signal = True
                    else:
                        # For early periods, use absolute volume threshold
                        if vol_1d_avg[i] > np.percentile(vol_1d[:i+1], 70) if i > 0 else False:
                            long_signal = True
            
            # Enter short on Donchian breakdown with volume confirmation
            short_signal = False
            if high_vol_regime:
                if price < donch_low[i] and vol_1d_avg[i] > 0:
                    if i >= 20:
                        vol_ma_20_current = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().iloc[i]
                        if vol_1d_avg[i] > 1.5 * vol_ma_20_current:
                            short_signal = True
                    else:
                        if vol_1d_avg[i] > np.percentile(vol_1d[:i+1], 70) if i > 0 else False:
                            short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on Donchian breakdown or low volatility regime
            exit_signal = False
            if price < donch_low[i] or atr_ratio_1w_aligned[i] < 0.8:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on Donchian breakout or low volatility regime
            exit_signal = False
            if price > donch_high[i] or atr_ratio_1w_aligned[i] < 0.8:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_VolatilityRegime"
timeframe = "4h"
leverage = 1.0