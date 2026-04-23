#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 1d ATR Regime Filter and Volume Spike
- Camarilla R3/S3 levels from prior 1d provide strong intraday support/resistance
- 1d ATR(14) / ATR(50) > 1.2 identifies high volatility regime for breakout validity
- Volume > 1.8x 20-period average confirms breakout momentum with moderate filtering
- Designed for 4h timeframe targeting 25-35 trades/year (100-140 over 4 years) to balance opportunity and fee drag
- Works in bull markets via breakouts with high volatility, in bear markets via mean reversion at strong levels during low volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d ATR(14) and ATR(50) for volatility regime filter
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    atr_50 = tr.rolling(window=50, min_periods=50).mean()
    atr_ratio = atr_14 / atr_50
    atr_ratio_values = atr_ratio.values
    
    # Align ATR ratio to 4h timeframe (completed 1d bar only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_values)
    
    # Volume confirmation: > 1.8x 20-period average (moderate filtering)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # ATR50 needs 50 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with volatility regime filter and volume spike
        # Long: price breaks above Camarilla R3 + high volatility regime (ATR_ratio>1.2) + volume spike
        # Short: price breaks below Camarilla S3 + high volatility regime (ATR_ratio>1.2) + volume spike
        long_signal = (close[i] > camarilla_r3_aligned[i] and 
                      atr_ratio_aligned[i] > 1.2 and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s3_aligned[i] and 
                       atr_ratio_aligned[i] > 1.2 and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: volatility regime weakening (ATR_ratio<1.0) or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: volatility weakening or price breaks below Camarilla S3
                if (atr_ratio_aligned[i] < 1.0 or 
                    close[i] < camarilla_s3_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: volatility weakening or price breaks above Camarilla R3
                if (atr_ratio_aligned[i] < 1.0 or 
                    close[i] > camarilla_r3_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0