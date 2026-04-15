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
    
    # Daily high/low for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily range (high - low) for volatility regime
    daily_range = high_1d - low_1d
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    
    # Volatility regime: high volatility when daily range > 1.5 * ATR(14)
    vol_regime = daily_range_aligned > (1.5 * atr_14_aligned)
    
    # 12-period RSI on close for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=12, min_periods=12).mean()
    avg_loss = pd.Series(loss).rolling(window=12, min_periods=12).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current > 1.3x median of last 24 periods
    vol_median = pd.Series(volume).rolling(window=24, min_periods=1).median()
    vol_threshold = 1.3 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(vol_threshold[i]) or 
            np.isnan(vol_regime[i]) or np.isnan(daily_range_aligned[i])):
            continue
        
        # Only trade in high volatility regimes
        if not vol_regime[i]:
            signals[i] = 0.0
            continue
        
        # Long: RSI oversold (< 30) + volume mean reversion
        if rsi_values[i] < 30 and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: RSI overbought (> 70) + volume mean reversion
        elif rsi_values[i] > 70 and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: RSI returns to neutral zone (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and rsi_values[i] >= 40) or
               (signals[i-1] == -0.25 and rsi_values[i] <= 60))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_VolRegime_RSI_MeanReversion"
timeframe = "12h"
leverage = 1.0