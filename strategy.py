#!/usr/bin/env python3
"""
4h_atr_breakout_1w_trend_volume_v1
Hypothesis: On 4h timeframe, use weekly ATR breakout combined with 1d EMA trend filter and volume confirmation. 
Breakout direction follows weekly trend (EMA50 > EMA200 = long bias, EMA50 < EMA200 = short bias). 
Enter long when price breaks above weekly ATR-based upper band with bullish trend and volume > 1.5x average. 
Enter short when price breaks below weekly ATR-based lower band with bearish trend and volume > 1.5x average. 
Exit on opposite band touch or trend reversal. Weekly ATR provides volatility-adaptive breakout levels 
that work in both trending and ranging markets. Volume confirms institutional participation. 
Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1w_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate weekly ATR-based bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly ATR-based bands (using weekly close as anchor)
    # Upper band: weekly close + 2.0 * ATR
    # Lower band: weekly close - 2.0 * ATR
    upper_band = close_1w + 2.0 * atr
    lower_band = close_1w - 2.0 * atr
    
    # Align to 4h timeframe (shifted by 1 week for look-ahead prevention)
    upper_band_4h = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_4h = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(ema200[i]) or 
            np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price touches lower band (mean reversion)
            if close[i] <= lower_band_4h[i]:
                exit_long = True
            # Exit if EMA50 crosses below EMA200 (trend reversal)
            elif ema50[i] < ema200[i] and ema50[i-1] >= ema200[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price touches upper band (mean reversion)
            if close[i] >= upper_band_4h[i]:
                exit_short = True
            # Exit if EMA50 crosses above EMA200 (trend reversal)
            elif ema50[i] > ema200[i] and ema50[i-1] <= ema200[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with bullish trend and volume confirmation
            long_entry = False
            if (close[i] > upper_band_4h[i] and close[i-1] <= upper_band_4h[i-1] and
                ema50[i] > ema200[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below lower band with bearish trend and volume confirmation
            short_entry = False
            if (close[i] < lower_band_4h[i] and close[i-1] >= lower_band_4h[i-1] and
                ema50[i] < ema200[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals