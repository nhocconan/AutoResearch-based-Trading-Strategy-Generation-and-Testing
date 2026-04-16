#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation and ATR filter.
# Long when price breaks above upper BB(20,2) AND 1d volume > 1.5x 20-period average AND 6h ATR(14) > 0.5*ATR(50) (expanding volatility).
# Short when price breaks below lower BB(20,2) AND 1d volume > 1.5x 20-period average AND 6h ATR(14) > 0.5*ATR(50).
# Uses discrete position size 0.25. BB squeeze identifies low volatility compression, breakout captures expansion.
# 1d volume filter ensures participation across higher timeframe, ATR filter avoids breakouts during low volatility.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_ma + 2 * bb_std).values
    bb_lower = (bb_ma - 2 * bb_std).values
    
    # === 6h Indicators: ATR(14) and ATR(50) for volatility filter ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean()
    atr_ratio = (atr_14 / atr_50).values  # > 0.5 indicates expanding volatility
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR(50), 20 for BB)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        vol_expanding = atr_ratio[i] > 0.5
        vol_spike = volume_spike_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band or volatility contracts
            if price <= bb_ma.iloc[i] or not vol_expanding:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band or volatility contracts
            if price >= bb_ma.iloc[i] or not vol_expanding:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper BB AND 1d volume spike AND volatility expanding
            if price > bb_up and vol_spike and vol_expanding:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower BB AND 1d volume spike AND volatility expanding
            elif price < bb_low and vol_spike and vol_expanding:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_BB_Squeeze_Breakout_1dVol_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0