# Theta: 4h Volume-Pressure Breakout with 1D ATR Filter
# Hypothesis: Breakouts above/below 20-period high/low on 4h with volume spike (>2x average) and 1D ATR > 0.8% of price
# work in bull/bear because volatility expansion precedes trends. Volume confirms institutional participation.
# Target: 20-40 trades/year per symbol. Max position size 0.30.
# Uses 1D ATR as regime filter to avoid low-volatility chop.

name = "Theta_4h_VolumePressure_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # 1D ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range and ATR on 1D data
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR as percentage of price (avoid zero division)
    atr_pct = np.where(df_1d['close'].values > 0, atr_1d / df_1d['close'].values, 0)
    atr_pct_aligned = align_htf_to_ltf(prices, df_1d, atr_pct)
    
    # Volatility filter: only trade when 1D ATR > 0.8% of price
    vol_filter = atr_pct_aligned > 0.008
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough for 20-period lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_pct_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 20-period high with volume spike and sufficient volatility
            if (close[i] > high_20[i] and 
                volume_spike[i] and 
                vol_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short: break below 20-period low with volume spike and sufficient volatility
            elif (close[i] < low_20[i] and 
                  volume_spike[i] and 
                  vol_filter[i]):
                signals[i] = -0.30
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 20-period low or volatility drops
            if (close[i] < low_20[i]) or (atr_pct_aligned[i] < 0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:
            # Short: exit if price breaks above 20-period high or volatility drops
            if (close[i] > high_20[i]) or (atr_pct_aligned[i] < 0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals