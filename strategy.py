#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action near 1d VWAP with volume surge and momentum confirmation
# Long when price crosses above 1d VWAP with rising momentum and volume spike
# Short when price crosses below 1d VWAP with falling momentum and volume spike
# Uses 1d VWAP as dynamic support/resistance that works in both trending and ranging markets
# Volume surge filters out weak breakouts, momentum confirms direction
# Designed for 4h timeframe to target 20-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for VWAP calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = typical_price_1d * volume_1d
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # Align 1d VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 4-period RSI for momentum confirmation (4h timeframe)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (15-period on 4h data)
    vol_ma15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_spike = volume > 2.0 * vol_ma15
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma15[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP + rising momentum (RSI > 50) + volume spike
            if (close[i] > vwap_1d_aligned[i] and 
                rsi[i] > 50 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP + falling momentum (RSI < 50) + volume spike
            elif (close[i] < vwap_1d_aligned[i] and 
                  rsi[i] < 50 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to VWAP or momentum divergence
            if position == 1:
                # Exit long on price below VWAP or momentum weakening
                if (close[i] < vwap_1d_aligned[i] or 
                    rsi[i] < 40):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short on price above VWAP or momentum strengthening
                if (close[i] > vwap_1d_aligned[i] or 
                    rsi[i] > 60):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Momentum_Volume_Spike"
timeframe = "4h"
leverage = 1.0