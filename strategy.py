#!/usr/bin/env python3
# 4h_daily_cci_trend_reversal_v1
# Hypothesis: 4h strategy using daily CCI for mean reversion in ranging markets + volume confirmation.
# Long: Daily CCI < -100 (oversold) AND price > 4h EMA20 (mild trend filter) AND volume > 1.3x 20-period average.
# Short: Daily CCI > 100 (overbought) AND price < 4h EMA20 AND volume > 1.3x 20-period average.
# Exit: Price crosses 4h EMA20 or daily CCI returns to [-20,20] range.
# Uses daily CCI for extreme conditions, 4h for execution, volume for confirmation.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_cci_trend_reversal_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 4h EMA20 for trend filter
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Get 1d data for CCI (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily CCI (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp = (high_1d + low_1d + close_1d) / 3.0
    # SMA of typical price
    tp_sma = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    md = np.where(md == 0, 0.001, md)
    # CCI
    cci = (tp - tp_sma) / (0.015 * md)
    
    # Align HTF CCI to LTF (completed 1d bar only)
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cci_aligned[i]) or np.isnan(ema20[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below EMA20 OR CCI returns to neutral range
            if close[i] < ema20[i] or abs(cci_aligned[i]) <= 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above EMA20 OR CCI returns to neutral range
            if close[i] > ema20[i] or abs(cci_aligned[i]) <= 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for mean reversion entry with volume confirmation
            oversold = cci_aligned[i] < -100
            overbought = cci_aligned[i] > 100
            
            long_entry = oversold and (close[i] > ema20[i]) and volume_confirmed
            short_entry = overbought and (close[i] < ema20[i]) and volume_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals