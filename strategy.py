#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_volatility_breakout_v1
# Volatility-based breakout using ATR expansion and price channels. In bull markets, 
# buy when price breaks above ATR-based resistance with expanding volatility. In bear
# markets, sell when price breaks below ATR-based support with volatility expansion.
# Uses volume confirmation to avoid false breakouts and ATR-based position sizing.
# Target: 20-40 trades/year per symbol for low friction and high win rate.
name = "4h_1d_volatility_breakout_v1"
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
    
    # Get 1d data for ATR calculation (more stable on higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR on 1d timeframe (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate dynamic channels using ATR (similar to Keltner)
    # Use 20-period SMA of close for center line
    close_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    upper_channel = close_ma + (2.0 * atr_1d_aligned)
    lower_channel = close_ma - (2.0 * atr_1d_aligned)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Volatility expansion filter: current ATR > 1.2 * 20-period ATR average
    atr_ma = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
    vol_expansion = atr_1d_aligned > (atr_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(atr_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check filters: volume confirmation and volatility expansion
        if not (vol_confirm[i] and vol_expansion[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above upper channel with volatility expansion
        if close[i] > upper_channel[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below lower channel with volatility expansion
        elif close[i] < lower_channel[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to middle (mean reversion tendency)
        elif abs(close[i] - close_ma[i]) < (0.5 * atr_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals