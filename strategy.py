#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI filter (14-period)
    close_series = pd.Series(close)
    rsi_14 = 100 - (100 / (1 + close_series.diff().clip(lower=0).rolling(14, min_periods=14).mean() / 
                              (-close_series.diff().clip(upper=0).abs().rolling(14, min_periods=14).mean())))
    rsi_14 = rsi_14.fillna(50).values
    
    # Daily high/low for volatility filter
    daily_high = get_htf_data(prices, '1d')['high'].values
    daily_low = get_htf_data(prices, '1d')['low'].values
    daily_range = daily_high - daily_low
    daily_atr = pd.Series(daily_range).rolling(14, min_periods=14).mean().values
    
    # Align daily ATR to 6h
    daily_atr_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), daily_atr)
    
    # 6-period RSI for momentum
    rsi_6 = 100 - (100 / (1 + pd.Series(close).diff().clip(lower=0).rolling(6, min_periods=6).mean() / 
                           (-pd.Series(close).diff().clip(upper=0).abs().rolling(6, min_periods=6).mean())))
    rsi_6 = rsi_6.fillna(50).values
    
    # Volume filter: current > 1.5x median of last 24 periods (4 days)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14[i]) or np.isnan(daily_atr_aligned[i]) or 
            np.isnan(rsi_6[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility regime: only trade when volatility is elevated
        vol_regime = daily_atr_aligned[i] > np.nanmedian(daily_atr_aligned[max(0, i-100):i+1])
        
        # Long: RSI14 oversold + RSI6 bullish divergence + volume + vol regime
        if (rsi_14[i] < 30 and rsi_6[i] > 50 and 
            volume[i] > vol_threshold[i] and vol_regime):
            signals[i] = 0.25
        
        # Short: RSI14 overbought + RSI6 bearish divergence + volume + vol regime
        elif (rsi_14[i] > 70 and rsi_6[i] < 50 and 
              volume[i] > vol_threshold[i] and vol_regime):
            signals[i] = -0.25
        
        # Exit: RSI returns to neutral zone
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and rsi_14[i] >= 50) or
               (signals[i-1] == -0.25 and rsi_14[i] <= 50))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_RSI_Divergence_Volume"
timeframe = "6h"
leverage = 1.0