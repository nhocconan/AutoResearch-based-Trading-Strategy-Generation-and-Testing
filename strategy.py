#!/usr/bin/env python3
# Hypothesis: 6h ATR-based volatility breakout with 1d trend filter and volume confirmation
# Long when price breaks above ATR-based upper band with 1d EMA50 uptrend and volume > 1.5x average
# Short when price breaks below ATR-based lower band with 1d EMA50 downtrend and volume > 1.5x average
# Exit when price reverses back to the 6-period average price (mean reversion)
# Uses volatility breakout to capture momentum bursts in both trending and ranging markets
# Designed to work in both bull and bear markets by filtering with daily trend
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_ATR_Volatility_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) for volatility bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-period average price for mean reversion exit
    avg_price = pd.Series(close).rolling(window=6, min_periods=6).mean().values
    
    # Volatility bands: ±2 * ATR from close
    upper_band = close + 2.0 * atr
    lower_band = close - 2.0 * atr
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ATR and EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_price[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, 1d EMA50 uptrend, volume spike
            if (close[i] > upper_band[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, 1d EMA50 downtrend, volume spike
            elif (close[i] < lower_band[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to average price (mean reversion)
            if close[i] <= avg_price[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to average price (mean reversion)
            if close[i] >= avg_price[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals