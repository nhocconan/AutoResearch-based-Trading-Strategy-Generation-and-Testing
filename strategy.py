#!/usr/bin/env python3
"""
1d_bollinger_bands_pullback_1w_trend_volume_v1
Hypothesis: On daily timeframe, buy pullbacks to Bollinger Bands lower band during weekly uptrend with volume confirmation.
Sell rallies to upper band during weekly downtrend with volume confirmation.
Uses mean reversion within trend context to capture swings in both bull and bear markets.
Targets 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_bollinger_bands_pullback_1w_trend_volume_v1"
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
    
    # Bollinger Bands (20, 2) on daily close
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on weekly close
    weekly_close = df_1w['close'].values
    weekly_close_s = pd.Series(weekly_close)
    ema50_1w = weekly_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1w = weekly_close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align to daily timeframe (shifted by 1 week to avoid look-ahead)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after Bollinger Bands warmup
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter from weekly: up if EMA50 > EMA200, down if EMA50 < EMA200
        trend_up = ema50_1w_aligned[i] > ema200_1w_aligned[i]
        trend_down = ema50_1w_aligned[i] < ema200_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price touches or crosses middle band (mean reversion complete)
            if close[i] >= bb_middle[i]:
                exit_long = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price touches or crosses middle band (mean reversion complete)
            if close[i] <= bb_middle[i]:
                exit_short = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at or below lower band, weekly uptrend, volume confirmation
            long_entry = (close[i] <= bb_lower[i]) and trend_up and vol_confirm
            
            # Short entry: price at or above upper band, weekly downtrend, volume confirmation
            short_entry = (close[i] >= bb_upper[i]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals