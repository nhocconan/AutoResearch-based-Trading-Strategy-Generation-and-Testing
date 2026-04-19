#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly volume confirmation and volatility regime filter.
# Long when price breaks above 20-day Donchian high AND weekly volume > 1.5x weekly average volume AND weekly ATR(7) < ATR(21) (low volatility regime)
# Short when price breaks below 20-day Donchian low AND weekly volume > 1.5x weekly average volume AND weekly ATR(7) < ATR(21)
# Exit when price crosses back through the Donchian midpoint
# Uses Donchian for trend following structure, weekly volume for confirmation, weekly ATR regime filter to avoid chop.
# Target: 10-25 trades/year per symbol.
name = "1d_Donchian_WeeklyVolume_VolatilityRegime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for regime filter and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate True Range components for weekly ATR
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr21 = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    # Align ATR arrays to daily timeframe
    atr7_aligned = align_htf_to_ltf(prices, df_1w, atr7)
    atr21_aligned = align_htf_to_ltf(prices, df_1w, atr21)
    
    # Get weekly average volume for confirmation
    vol_ma_1w = pd.Series(df_1w['volume']).rolling(window=10, min_periods=10).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr7_aligned[i]) or np.isnan(atr21_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr7_val = atr7_aligned[i]
        atr21_val = atr21_aligned[i]
        vol_ma = vol_ma_1w_aligned[i]
        vol = volume[i]
        upper = high_roll[i]
        lower = low_roll[i]
        mid = donchian_mid[i]
        
        # Regime filter: only trade in low volatility (weekly ATR7 < ATR21)
        vol_regime = atr7_val < atr21_val
        
        if position == 0:
            # Long entry: break above upper band + volume spike + low vol regime
            if price > upper and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band + volume spike + low vol regime
            elif price < lower and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals