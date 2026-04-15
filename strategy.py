#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Bollinger Bands: 20-period, 2 std
    daily12 = get_htf_data(prices, '12h')
    close_12h = daily12['close'].values
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    upper_12h_aligned = align_htf_to_ltf(prices, daily12, upper_band)
    lower_12h_aligned = align_htf_to_ltf(prices, daily12, lower_band)
    
    # 1d ATR(14) for volatility filter
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume threshold: 1.8x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median()
    vol_threshold = 1.8 * vol_median
    
    # Bollinger Band width for regime filter
    bb_width = (upper_12h_aligned - lower_12h_aligned) / sma_20  # Will align later
    bb_width_series = pd.Series(bb_width)
    bb_width_aligned = align_htf_to_ltf(prices, daily12, bb_width_series.values)
    bb_width_median = pd.Series(bb_width_aligned).rolling(window=100, min_periods=100).median()
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or
            np.isnan(atr_14d_aligned[i]) or np.isnan(vol_threshold[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(bb_width_median[i])):
            continue
        
        # Regime filter: avoid extremely low volatility (squeeze) and extremely high volatility
        vol_regime = (bb_width_aligned[i] > 0.5 * bb_width_median[i]) and (bb_width_aligned[i] < 3.0 * bb_width_median[i])
        
        # Volatility filter: avoid extremes (0.6x to 2.5x of ATR median)
        atr_median = pd.Series(atr_14d_aligned).rolling(window=100, min_periods=100).median()
        vol_filter = (atr_14d_aligned[i] > 0.6 * atr_median[i]) and (atr_14d_aligned[i] < 2.5 * atr_median[i])
        
        # Long: Price breaks above upper Bollinger Band + volume spike + volatility filter
        if (close[i] > upper_12h_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_regime and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price breaks below lower Bollinger Band + volume spike + volatility filter
        elif (close[i] < lower_12h_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_regime and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price returns inside Bollinger Bands
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper_12h_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > lower_12h_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_12h_BollingerBreakout_Vol1.8x_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0