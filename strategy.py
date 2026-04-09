#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# - Primary timeframe: 6h for lower fee drag and better signal quality
# - HTF: 1d for trend direction (price > SMA50 = bull, price < SMA50 = bear)
# - Entry: Donchian(20) breakout in direction of 1d trend with volume > 1.5x 20-period average
# - Exit: ATR(10) trailing stop (2.0x ATR from extreme) or opposite Donchian breakout
# - Position size: 0.25 (25% of capital) to control drawdown in volatile markets
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years)
# - Donchian breakouts capture trends, ATR stop manages risk, volume filter reduces false breakouts
# - 1d regime filter avoids counter-trend trades in strong markets

name = "6h_1d_donchian_atr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d SMA50 for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(10) for volatility and stoploss
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # 1d Volume > 1.5x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_20)
    
    # Align 1d indicators to 6h
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR opposite Donchian breakout
            if (low[i] <= highest_since_entry - (2.0 * atr_1d_aligned[i]) or
                low[i] <= lowest_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR opposite Donchian breakout
            if (high[i] >= lowest_since_entry + (2.0 * atr_1d_aligned[i]) or
                high[i] >= highest_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Determine 1d trend regime
            bull_regime = close[i] > sma50_1d_aligned[i]   # Price above 1d SMA50 = bull
            bear_regime = close[i] < sma50_1d_aligned[i]   # Price below 1d SMA50 = bear
            
            # Look for Donchian breakout with volume confirmation and regime alignment
            if (high[i] >= highest_20[i] and    # Break above upper Donchian
                bull_regime and                 # Only long in bull regime
                volume_spike_1d_aligned[i]):    # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= lowest_20[i] and    # Break below lower Donchian
                  bear_regime and               # Only short in bear regime
                  volume_spike_1d_aligned[i]):  # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals