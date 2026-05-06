#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Supertrend for trend direction and 1d ATR for volatility filtering
# - Uses 1w Supertrend (ATR=10, multiplier=3) to determine primary trend direction
# - Uses 1d ATR to filter entries during normal volatility periods (avoid extreme volatility)
# - Enters long when price retraces to 1w Supertrend line during uptrend with volume confirmation
# - Enters short when price retraces to 1w Supertrend line during downtrend with volume confirmation
# - Exits when price moves 1.5x ATR away from Supertrend line or trend reverses
# - Designed to capture trend retracement entries in both bull and bear markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Supertrend_ATR_Retracement"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1w Supertrend (ATR=10, multiplier=3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(10)
    atr_period = 10
    atr_1w = np.zeros_like(tr)
    atr_1w[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr_1w[i] = (atr_1w[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    multiplier = 3
    hl_avg = (high_1w + low_1w) / 2
    upper_band = hl_avg + (multiplier * atr_1w)
    lower_band = hl_avg - (multiplier * atr_1w)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Calculate 1d ATR for volatility filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First period
    
    # ATR(14)
    atr_period_1d = 14
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[atr_period_1d-1] = np.mean(tr_1d[:atr_period_1d])
    for i in range(atr_period_1d, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * (atr_period_1d-1) + tr_1d[i]) / atr_period_1d
    
    # Align 1w indicators to 12h timeframe
    supertrend_12h = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_12h = align_htf_to_ltf(prices, df_1w, direction)
    upper_band_12h = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Align 1d ATR to 12h timeframe
    atr_1d_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filters (12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Moderate volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(supertrend_12h[i]) or np.isnan(direction_12h[i]) or 
            np.isnan(atr_1d_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for retracement to Supertrend line during trending market
            # Avoid extreme volatility (ATR > 3x median ATR)
            median_atr = np.nanmedian(atr_1d_12h[max(0, i-50):i+1])
            normal_volatility = atr_1d_12h[i] < (3 * median_atr) if not np.isnan(median_atr) else True
            
            if normal_volatility:
                # Long: price retraces to Supertrend line during uptrend with volume confirmation
                if (direction_12h[i] == 1 and 
                    close[i] <= supertrend_12h[i] * 1.005 and  # Near Supertrend (allow 0.5% above)
                    close[i] >= supertrend_12h[i] * 0.995 and  # Near Supertrend (allow 0.5% below)
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price retraces to Supertrend line during downtrend with volume confirmation
                elif (direction_12h[i] == -1 and 
                      close[i] >= supertrend_12h[i] * 0.995 and  # Near Supertrend (allow 0.5% below)
                      close[i] <= supertrend_12h[i] * 1.005 and  # Near Supertrend (allow 0.5% above)
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price moves 1.5x ATR above Supertrend OR trend reverses to downtrend
            if (close[i] > supertrend_12h[i] + (1.5 * atr_1d_12h[i]) or 
                direction_12h[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price moves 1.5x ATR below Supertrend OR trend reverses to uptrend
            if (close[i] < supertrend_12h[i] - (1.5 * atr_1d_12h[i]) or 
                direction_12h[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals