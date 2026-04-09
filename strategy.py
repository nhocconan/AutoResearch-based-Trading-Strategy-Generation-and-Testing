#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels with volume confirmation and session filter
# Camarilla pivots from 4h provide intraday support/resistance that work in ranging markets
# Volume confirmation (current 1h volume > 1.3x 20-period average) filters false breakouts
# Session filter (08-20 UTC) avoids low-liquidity Asian session noise
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Position size fixed at 0.20 to minimize fee churn and control drawdown

name = "1h_4h_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for 4h
    # Based on previous 4h bar's OHLC
    camarilla_h5 = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_h4 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_h3 = close_4h + (high_4h - low_4h) * 1.1 / 6
    camarilla_l3 = close_4h - (high_4h - low_4h) * 1.1 / 6
    camarilla_l4 = close_4h - (high_4h - low_4h) * 1.1 / 4
    camarilla_l5 = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (shifted by 1 for completed bar)
    h5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l5)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h5_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(l5_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.3x average 1h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on retracement to H3 level or below
            if close[i] < h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit on retracement to L3 level or above
            if close[i] > l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Mean reversion trading at extreme Camarilla levels with volume confirmation
            # Short at H4/H5, Long at L4/L5
            if volume_confirmed:
                if close[i] > h4_aligned[i]:
                    position = -1
                    signals[i] = -0.20
                elif close[i] < l4_aligned[i]:
                    position = 1
                    signals[i] = 0.20
    
    return signals