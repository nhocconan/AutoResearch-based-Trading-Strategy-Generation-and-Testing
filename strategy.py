#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Volume Weighted Average Price (VWAP) as dynamic support/resistance.
# Price tends to revert to daily VWAP in ranging markets and break with volume in trending markets.
# Long when price crosses above VWAP with rising volume and RSI>50; short when crosses below VWAP with rising volume and RSI<50.
# Exit on opposite VWAP touch or RSI reversal. Uses 1-day VWAP to avoid noise and capture institutional levels.
# Designed for low frequency (15-25 trades/year) to minimize fee drag while capturing meaningful moves.
name = "6h_VWAP_MeanReversion_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Typical price and VWAP for each day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align VWAP to 6h timeframe (uses prior day's VWAP)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP with bullish momentum and volume
            if (close[i] > vwap_aligned[i] and 
                close[i-1] <= vwap_aligned[i-1] and  # crossed above this bar
                rsi[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with bearish momentum and volume
            elif (close[i] < vwap_aligned[i] and 
                  close[i-1] >= vwap_aligned[i-1] and  # crossed below this bar
                  rsi[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches VWAP from above or RSI turns bearish
            if (close[i] < vwap_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches VWAP from below or RSI turns bullish
            if (close[i] > vwap_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals