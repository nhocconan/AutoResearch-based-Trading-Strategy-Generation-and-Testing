#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based volatility breakout with 1d EMA200 trend filter and volume confirmation.
# Uses daily EMA200 for trend filter and ATR(14) breakout from 4h close, aligned to 4h.
# Volume spike confirms breakout strength. Designed to capture strong trends with low turnover.
# Target: 15-40 trades/year to stay within optimal range for 4h timeframe.
# Volatility breakouts work in both bull and bear markets by capturing expansion phases.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate ATR(14) on 4h for volatility breakout levels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility breakout levels: close +/- 1.5 * ATR
    upper_break = close + (1.5 * atr14)
    lower_break = close - (1.5 * atr14)
    
    # Volume filter: current volume > 2.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need EMA200(1d) + ATR14(4h) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_4h[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.5x average (strict to reduce trades)
        volume_filter = volume[i] > (2.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema200_4h[i]
        price_below_ema = close[i] < ema200_4h[i]
        
        # Price relative to volatility breakout levels
        price_above_upper = close[i] > upper_break[i]
        price_below_lower = close[i] < lower_break[i]
        
        if position == 0:
            # Long: Price breaks above upper volatility band with volume and above 1d EMA200
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower volatility band with volume and below 1d EMA200
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below lower volatility band OR below 1d EMA200
            if (close[i] < lower_break[i]) or (close[i] < ema200_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above upper volatility band OR above 1d EMA200
            if (close[i] > upper_break[i]) or (close[i] > ema200_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ATRBreakout_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0