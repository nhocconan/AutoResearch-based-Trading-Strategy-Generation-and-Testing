#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Bollinger Bands for trend direction and 1d ATR for volatility filter.
# Long when price > 4h BB upper band, ATR ratio > 1.2, and volume > 1.2x 20-period average.
# Short when price < 4h BB lower band, ATR ratio > 1.2, and volume > 1.2x 20-period average.
# Uses discrete position size 0.20 to limit risk and reduce churn.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).
# Works in bull/bear by capturing volatility expansion breakouts with volume confirmation.
name = "1h_BB4h_ATR1d_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_4h = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_4h = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_4h + (bb_std * std_4h)
    bb_lower = sma_4h - (bb_std * std_4h)
    
    # Align 4h Bollinger Bands to 1h
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    
    # Get 1d data for ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14)
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Align 1d ATR to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # ATR ratio: current 1h ATR / 1d ATR (volatility expansion filter)
    tr_1h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_1h[0] = 0
    atr_1h = pd.Series(tr_1h).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_ratio = np.where(atr_1d_aligned > 0, atr_1h / atr_1d_aligned, 0)
    
    # Volume confirmation: current volume > 1.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, atr_period, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        atr_ratio_val = atr_ratio[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volatility expansion filter
        vol_expansion = atr_ratio_val > 1.2
        
        # Volume confirmation
        volume_confirmed = vol > 1.2 * vol_ma
        
        if position == 0:
            # Enter long if price above 4h BB upper, volatility expansion, and volume confirmation
            if price > bb_upper_val and vol_expansion and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short if price below 4h BB lower, volatility expansion, and volume confirmation
            elif price < bb_lower_val and vol_expansion and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below 4h BB middle or volatility contracts
            bb_middle_aligned = sma_4h  # Will align below
            bb_middle_aligned = align_htf_to_ltf(prices, df_4h, sma_4h)
            if price < bb_middle_aligned[i] or atr_ratio_val < 0.8:  # Volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price crosses above 4h BB middle or volatility contracts
            bb_middle_aligned = align_htf_to_ltf(prices, df_4h, sma_4h)
            if price > bb_middle_aligned[i] or atr_ratio_val < 0.8:  # Volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals