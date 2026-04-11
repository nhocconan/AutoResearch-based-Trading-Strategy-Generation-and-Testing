#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Momentum_v1
Hypothesis: Combines daily Camarilla pivot breakouts with 4-hour momentum confirmation (RSI divergence) and volume filtering.
Designed for low trade frequency (<25/year) with high-probability setups in both bull and bear markets by requiring
confluence of price action, momentum, and volume. Uses discrete position sizing to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from daily data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Momentum confirmation: RSI not overextended
        rsi_not_overbought = rsi_values[i] < 70
        rsi_not_oversold = rsi_values[i] > 30
        
        # Breakout conditions using Camarilla R4/S4 levels
        breakout_up = close[i] > camarilla_r4_aligned[i]  # Break above R4
        breakdown_down = close[i] < camarilla_s4_aligned[i]  # Break below S4
        
        # Entry conditions: require volume and momentum confirmation
        long_entry = breakout_up and volume_filter and rsi_not_overbought
        short_entry = breakdown_down and volume_filter and rsi_not_oversold
        
        # Exit conditions: return to opposite Camarilla level or momentum divergence
        long_exit = (close[i] < camarilla_s4_aligned[i]) or (rsi_values[i] > 75)
        short_exit = (close[i] > camarilla_r4_aligned[i]) or (rsi_values[i] < 25)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals