# 4h_12h_Camarilla_Pivot_Volume_Filter_v1
# Hypothesis: Camarilla pivot levels on 12h act as strong support/resistance in both bull and bear markets.
# Long when price touches L3 with volume confirmation; short when price touches H3 with volume confirmation.
# Uses 12h Camarilla levels (calculated from prior 12h bar) for structure, volume spike for confirmation.
# Works in trending markets (pullbacks to pivot levels) and ranging markets (oscillations between H3/L3).
# Volume filter reduces false breakouts. Target: 20-50 trades/year on 4h timeframe.

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
    
    # Load 12h data ONCE for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels for each 12h bar: based on prior bar's OHLC
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    # We use prior bar to avoid look-ahead
    h12 = df_12h['high'].values
    l12 = df_12h['low'].values
    c12 = df_12h['close'].values
    
    # Prior bar values (shifted by 1)
    h12_prev = np.concatenate([[np.nan], h12[:-1]])
    l12_prev = np.concatenate([[np.nan], l12[:-1]])
    c12_prev = np.concatenate([[np.nan], c12[:-1]])
    
    # Camarilla levels
    H3 = c12_prev + (h12_prev - l12_prev) * 1.1 / 4
    L3 = c12_prev - (h12_prev - l12_prev) * 1.1 / 4
    
    # Align to 4h timeframe (will only update after 12h bar closes)
    H3_4h = align_htf_to_ltf(prices, df_12h, H3)
    L3_4h = align_htf_to_ltf(prices, df_12h, L3)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if Camarilla levels are not available (NaN)
        if np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price touches L3 with volume confirmation
            if price <= L3_4h[i] and volume_spike[i]:
                position = 1
                signals[i] = position_size
            # Enter short: price touches H3 with volume confirmation
            elif price >= H3_4h[i] and volume_spike[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches midpoint (H3+L3)/2 or touches H3
            midpoint = (H3_4h[i] + L3_4h[i]) / 2
            if price >= midpoint or price >= H3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches midpoint or touches L3
            midpoint = (H3_4h[i] + L3_4h[i]) / 2
            if price <= midpoint or price <= L3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Camarilla_Pivot_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0