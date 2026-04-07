#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Close vs Weekly VWAP with Volume Confirmation
# Hypothesis: Price relative to weekly VWAP indicates institutional bias.
# In weekly uptrend (weekly VWAP rising), go long when daily close > weekly VWAP and volume > average.
# In weekly downtrend (weekly VWAP falling), go short when daily close < weekly VWAP and volume > average.
# Uses weekly timeframe for trend bias and daily for entry/exit.
# Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
name = "1d_close_vs_weekly_vwap_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP: cumulative (price * volume) / cumulative volume
    typical_price_weekly = (df_weekly['high'].values + df_weekly['low'].values + df_weekly['close'].values) / 3.0
    pv_weekly = typical_price_weekly * df_weekly['volume'].values
    cum_pv = np.cumsum(pv_weekly)
    cum_vol = np.cumsum(df_weekly['volume'].values)
    vwap_weekly = cum_pv / cum_vol
    vwap_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vwap_weekly)
    
    # Weekly VWAP slope for trend detection (3-period change)
    vwap_slope = np.diff(vwap_weekly_aligned, prepend=vwap_weekly_aligned[0])
    vwap_rising = vwap_slope > 0
    vwap_falling = vwap_slope < 0
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(vwap_weekly_aligned[i]) or np.isnan(vwap_rising[i]) or 
            np.isnan(vwap_falling[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP or volume dries up
            if close[i] < vwap_weekly_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP or volume dries up
            if close[i] > vwap_weekly_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long when: weekly VWAP rising AND price above VWAP
                if vwap_rising[i] and close[i] > vwap_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short when: weekly VWAP falling AND price below VWAP
                elif vwap_falling[i] and close[i] < vwap_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals