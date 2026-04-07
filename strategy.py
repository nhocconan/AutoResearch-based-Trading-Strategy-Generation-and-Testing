#!/usr/bin/env python3
"""
12h_momentum_reversal_1d_vwap_v1
Hypothesis: On 12-hour timeframe, use daily VWAP as dynamic support/resistance with momentum confirmation. Long when price crosses above VWAP with rising momentum, short when price crosses below VWAP with falling momentum. Uses 1-day trend filter and volume confirmation to avoid whipsaws. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while capturing both mean-reversion and momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_momentum_reversal_1d_vwap_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily VWAP (typical price * volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Calculate daily EMA(20) for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily VWAP and EMA(20) to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Momentum indicator: ROC(5) on 12h timeframe
    roc_period = 5
    roc = np.zeros(n)
    for i in range(roc_period, n):
        if close[i - roc_period] != 0:
            roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period]
    
    # Volume filter: 24-period average on 12h timeframe (equivalent to ~12 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(24, 50), n):
        # Skip if data not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price falls below VWAP with weakening momentum
            if close[i] < vwap_aligned[i] and roc[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above VWAP with strengthening momentum
            if close[i] > vwap_aligned[i] and roc[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price crosses above VWAP with positive momentum and bullish trend
                if (close[i] > vwap_aligned[i] and close[i-1] <= vwap_aligned[i-1] and 
                    roc[i] > 0.005 and ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below VWAP with negative momentum and bearish trend
                elif (close[i] < vwap_aligned[i] and close[i-1] >= vwap_aligned[i-1] and 
                      roc[i] < -0.005 and ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals