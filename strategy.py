#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day volume-weighted average price (VWAP) as dynamic support/resistance,
# combined with 4-hour RSI for momentum confirmation and volume spike for entry timing.
# VWAP provides institutional reference levels that work in both trending and ranging markets.
# RSI filters for overbought/oversold conditions, while volume spikes confirm institutional interest.
# Target: 20-40 trades/year per symbol with disciplined entries.
name = "4h_VWAP_RSI_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily VWAP calculation (typical price * volume) / cumulative volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate VWAP for each daily bar
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_values = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap_values.values
    
    # Align daily VWAP to 4h timeframe (1-day VWAP stays constant throughout the day)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # 4-hour RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above VWAP, RSI < 50 (not overbought), with volume spike
            if (close[i] > vwap_aligned[i] and 
                rsi[i] < 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP, RSI > 50 (not oversold), with volume spike
            elif (close[i] < vwap_aligned[i] and 
                  rsi[i] > 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below VWAP or RSI becomes overbought
            if (close[i] < vwap_aligned[i]) or (rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above VWAP or RSI becomes oversold
            if (close[i] > vwap_aligned[i]) or (rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals