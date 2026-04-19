#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour weekly EMA50 trend with daily ATR volatility filter and volume confirmation.
# Long when: Price closes above weekly EMA50, daily ATR > 1.2x 20-period average, volume > 1.5x 20-period average
# Short when: Price closes below weekly EMA50, daily ATR > 1.2x 20-period average, volume > 1.5x 20-period average
# Exit when: Price crosses back through weekly EMA50
# Weekly EMA50 filters long-term trend, ATR ensures sufficient volatility, volume confirms momentum.
# Target: 20-30 trades/year per symbol. Works in bull (buy pullbacks) and bear (sell rallies).
name = "12h_WeeklyEMA50_ATR_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Daily data for ATR and volume filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ATR 20-period average for volatility filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily volume 20-period average for volume filter
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA50 to 12H timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align daily ATR and volume averages to 12H timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_ma_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        atr = atr_14_aligned[i]
        atr_ma = atr_ma_20_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        
        if position == 0:
            # Long entry: Price above weekly EMA50, sufficient volatility, volume spike
            if (price > ema50 and atr > 1.2 * atr_ma and volume[i] > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price below weekly EMA50, sufficient volatility, volume spike
            elif (price < ema50 and atr > 1.2 * atr_ma and volume[i] > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below weekly EMA50
            if price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above weekly EMA50
            if price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals