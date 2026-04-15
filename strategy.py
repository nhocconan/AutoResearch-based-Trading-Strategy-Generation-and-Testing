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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    daily_tr = np.maximum(tr1, np.maximum(tr2, tr3))
    daily_atr = pd.Series(daily_tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR to 4h timeframe
    daily_atr_4h = align_htf_to_ltf(prices, df_1d, daily_atr)
    
    # Calculate 4h ATR(14) for stop loss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_atr_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: High volatility environment (daily ATR > 2% of price)
        vol_regime = daily_atr_4h[i] > 0.02 * close[i]
        
        # Entry conditions:
        # 1. 4h price breaks above recent 20-period high with volume confirmation → long
        # 2. 4h price breaks below recent 20-period low with volume confirmation → short
        # 3. Only trade in high volatility regimes (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Calculate 20-period high/low for breakout levels
        if i >= 20:
            high_20 = np.max(high[i-20:i])
            low_20 = np.min(low[i-20:i])
            
            # Long conditions: 4h breakout above 20-period high
            if (close[i] > high_20 and              # Break above 20-period high
                volume_ratio[i] > 1.5 and           # Volume confirmation
                vol_regime):                        # High volatility regime
                signals[i] = 0.25
                
            # Short conditions: 4h breakdown below 20-period low
            elif (close[i] < low_20 and             # Break below 20-period low
                  volume_ratio[i] > 1.5 and         # Volume confirmation
                  vol_regime):                      # High volatility regime
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Volatility_Breakout_Volume_Regime_Filter"
timeframe = "4h"
leverage = 1.0