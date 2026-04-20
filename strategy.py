#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE for HTF regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA(50) for trend regime
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h price data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h ATR(14) for stoploss
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in HTF indicators
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        atr_1d_val = atr_14_1d_aligned[i]
        atr_4h_val = atr_14_4h[i]
        price = close[i]
        
        # Volatility regime: only trade when volatility is below median (calm markets)
        vol_regime = atr_1d_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 50)
        
        if position == 0:
            # Long: price above daily EMA50 (uptrend), calm volatility
            if price > ema_50_val and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA50 (downtrend), calm volatility
            elif price < ema_50_val and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA50 or volatility spikes
            if price < ema_50_val or atr_4h_val > 2.0 * atr_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA50 or volatility spikes
            if price > ema_50_val or atr_4h_val > 2.0 * atr_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA50_1dTrend_VolRegime_ATRStop"
timeframe = "4h"
leverage = 1.0