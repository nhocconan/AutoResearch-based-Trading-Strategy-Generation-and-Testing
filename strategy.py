#!/usr/bin/env python3
"""
1h_Camillo_Momentum_VolumeRegime
Hypothesis: 1h momentum breakouts confirmed by 4h trend (EMA34) and volume regime (ATR-based) with session filter (08-20 UTC). Designed for 15-25 trades/year to avoid fee drag.
"""

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
    
    # 4h EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close']
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # ATR(14) for volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_regime = atr > atr_ma  # High volatility regime
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_reg = vol_regime[i]
        sess = session_filter[i]
        
        if position == 0:
            if sess and vol_reg:
                # Momentum breakout: price > EMA and rising
                if price > ema_trend and close[i] > close[i-1]:
                    signals[i] = 0.20
                    position = 1
                elif price < ema_trend and close[i] < close[i-1]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price crosses below EMA or low volatility
            if price < ema_trend or not vol_reg:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price crosses above EMA or low volatility
            if price > ema_trend or not vol_reg:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camillo_Momentum_VolumeRegime"
timeframe = "1h"
leverage = 1.0