#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day momentum filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (reversal signals)
# 1-day RSI momentum filter ensures we trade with higher timeframe momentum
# Volume confirmation filters out low-conviction moves
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag
# Works in bull/bear: Williams %R captures reversals in ranging markets,
# momentum filter avoids counter-trend trades in strong trends

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop for momentum filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day RSI (14) for momentum filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Williams %R (14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Williams %R and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Momentum filter: only take signals aligned with 1-day RSI
        bullish_momentum = rsi_1d_aligned[i] > 50
        bearish_momentum = rsi_1d_aligned[i] < 50
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with bullish momentum and volume
            if williams_r[i] < -80 and bullish_momentum and volume_confirm[i]:
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) with bearish momentum and volume
            elif williams_r[i] > -20 and bearish_momentum and volume_confirm[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or momentum shifts
            if williams_r[i] > -50 or not bullish_momentum:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or momentum shifts
            if williams_r[i] < -50 or not bearish_momentum:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_1dRSI_Momentum_Volume"
timeframe = "4h"
leverage = 1.0