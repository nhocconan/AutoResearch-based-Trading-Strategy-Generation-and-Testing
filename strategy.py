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
    
    # Weekly SMA(34) for long-term trend
    weekly = get_htf_data(prices, '1w')
    close_w = weekly['close'].values
    sma_34w = pd.Series(close_w).rolling(window=34, min_periods=34).mean().values
    sma_34w_aligned = align_htf_to_ltf(prices, weekly, sma_34w)
    
    # Daily ATR(14) for volatility
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Daily ATR mean for regime filter
    atr_mean = pd.Series(atr_14d_aligned).rolling(window=30, min_periods=30).mean()
    
    signals = np.zeros(n)
    
    for i in range(34, n):
        if (np.isnan(sma_34w_aligned[i]) or np.isnan(atr_14d_aligned[i]) or 
            np.isnan(atr_mean[i])):
            continue
        
        # Trend filter: price above/below weekly SMA34
        above_trend = close[i] > sma_34w_aligned[i]
        below_trend = close[i] < sma_34w_aligned[i]
        
        # Volatility filter: avoid low volatility (choppy) markets
        vol_filter = atr_14d_aligned[i] > 0.8 * atr_mean[i]
        
        # Entry conditions with volume confirmation
        if above_trend and vol_filter and volume[i] > 1.2 * pd.Series(volume).rolling(20, min_periods=20).mean()[i]:
            signals[i] = 0.30
        elif below_trend and vol_filter and volume[i] > 1.2 * pd.Series(volume).rolling(20, min_periods=20).mean()[i]:
            signals[i] = -0.30
        else:
            # Exit when price crosses weekly SMA34 or volatility drops
            if i > 0 and ((signals[i-1] > 0 and close[i] < sma_34w_aligned[i]) or
                         (signals[i-1] < 0 and close[i] > sma_34w_aligned[i]) or
                         atr_14d_aligned[i] < 0.5 * atr_mean[i]):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklySMA34_VolumeFilter_ATRRegime"
timeframe = "1d"
leverage = 1.0