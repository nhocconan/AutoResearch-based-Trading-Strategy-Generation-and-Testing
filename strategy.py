#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA200 + volume confirmation.
# Long when Williams %R < -80 (oversold), price > EMA200, volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought), price < EMA200, volume > 1.5x 20-period average.
# Uses discrete position size 0.25 to minimize churn. Designed for 12h timeframe to capture
# mean-reversion in both bull and bear markets while avoiding whipsaws.
# Target: 25-35 trades/year per symbol (~100-140 total over 4 years).
name = "12h_WilliamsR_EMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 220:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d EMA200 to 12h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14)  # Ensure EMA200 and Williams %R are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        wr = williams_r[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if oversold, price above EMA200, and volume confirmation
            if wr < -80 and price > ema_200_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if overbought, price below EMA200, and volume confirmation
            elif wr > -20 and price < ema_200_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R crosses above -50 or price crosses below EMA200
            if wr > -50 or price < ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R crosses below -50 or price crosses above EMA200
            if wr < -50 or price > ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals