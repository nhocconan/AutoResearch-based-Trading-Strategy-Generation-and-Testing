#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily (1d) price closes above/below Weekly (1w) VWAP with volume confirmation
# Uses weekly VWAP as dynamic support/resistance, filters by daily close direction
# Volume spike confirms institutional participation. Works in bull/bear via mean reversion
# around weekly VWAP (institutional anchor). Target: 15-25 trades/year.
name = "1d_WeeklyVWAP_Bounce_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly VWAP: cumulative (price * volume) / cumulative volume
    # Use typical price for VWAP calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vp = typical_price * df_1w['volume']
    cum_vp = vp.cumsum()
    cum_vol = df_1w['volume'].cumsum()
    vwap = (cum_vp / cum_vol).values
    
    # Align weekly VWAP to daily timeframe
    vwap_daily = align_htf_to_ltf(prices, df_1w, vwap)
    
    # 20-day volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(vwap_daily[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-day average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Price crosses above weekly VWAP with volume spike
            if close[i] > vwap_daily[i] and close[i-1] <= vwap_daily[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below weekly VWAP with volume spike
            elif close[i] < vwap_daily[i] and close[i-1] >= vwap_daily[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below weekly VWAP
            if close[i] < vwap_daily[i] and close[i-1] >= vwap_daily[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above weekly VWAP
            if close[i] > vwap_daily[i] and close[i-1] <= vwap_daily[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals