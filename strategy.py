# 4H MULTI-TIMEFRAME PIVOT BREAKOUT WITH VOLUME FILTER
# Hypothesis: 4H Camarilla pivot breakouts from daily levels work across market regimes.
# Uses 1D Camarilla pivot levels (R1/S1) as breakout triggers, confirmed by volume spikes.
# Works in bull (breaks above R1) and bear (breaks below S1) with volume confirmation.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years) to avoid fee drag.
# Edge: Pivot levels act as institutional support/resistance; breakouts with volume indicate
# genuine institutional interest, reducing false breakouts.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get DAILY data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily data
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Align daily pivot levels to 4H timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 4H ATR for volatility filter and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation
            if close[i] > r1_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation
            elif close[i] < s1_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below S1 (opposite level) or ATR trailing stop
            if close[i] < s1_aligned[i] or close[i] < high_since_entry - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Update highest high since entry for trailing stop
                if i == start_idx or position != 1:
                    high_since_entry = close[i]
                else:
                    high_since_entry = max(high_since_entry, close[i])
        
        elif position == -1:
            # Short exit: price crosses above R1 (opposite level) or ATR trailing stop
            if close[i] > r1_aligned[i] or close[i] > low_since_entry + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Update lowest low since entry for trailing stop
                if i == start_idx or position != -1:
                    low_since_entry = close[i]
                else:
                    low_since_entry = min(low_since_entry, close[i])
    
    return signals

name = "4H_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0