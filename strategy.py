#!/usr/bin/env python3
"""
12h_volatility_breakout_1w_trend_v1
Hypothesis: On 12h timeframe, trade breakouts of weekly Bollinger Bands with volume confirmation.
In bull markets, breakouts above upper band continue; in bear markets, breakouts below lower band continue.
Weekly timeframe filters noise, volume confirms institutional interest, Bollinger Bands adapt to volatility.
Targets 12-37 trades/year to minimize fee drag while capturing major moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_breakout_1w_trend_v1"
timeframe = "12h"
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
    
    # Calculate ATR for stop loss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Bollinger Bands (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on weekly close
    bb_period = 20
    bb_std = 2.0
    bb_middle = pd.Series(df_1w['close'].values).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(df_1w['close'].values).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + bb_std * bb_std_dev
    bb_lower = bb_middle - bb_std * bb_std_dev
    
    # Align to 12h timeframe (shifted by 1 week to avoid look-ahead)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or atr[i] <= 0 or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on breakdown below middle band with volume (trend exhaustion)
            if close[i] < bb_middle_aligned[i] and vol_confirm:
                exit_long = True
            # Stop loss: 2x ATR below entry (approximated by recent low)
            elif i >= 20 and close[i] < np.min(low[i-19:i+1]) - 2.0 * atr[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on break above middle band with volume (trend exhaustion)
            if close[i] > bb_middle_aligned[i] and vol_confirm:
                exit_short = True
            # Stop loss: 2x ATR above entry (approximated by recent high)
            elif i >= 20 and close[i] > np.max(high[i-19:i+1]) + 2.0 * atr[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Bollinger Band with volume confirmation
            long_entry = close[i] > bb_upper_aligned[i] and vol_confirm
            
            # Short entry: price breaks below lower Bollinger Band with volume confirmation
            short_entry = close[i] < bb_lower_aligned[i] and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals