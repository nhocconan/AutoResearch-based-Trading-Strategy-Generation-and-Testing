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
    
    # Get weekly HTF data once before loop (1w for 1d primary timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly ATR(14) for volatility regime filter
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]]))
    tr3 = np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF weekly ATR to daily timeframe
    atr_14_daily = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Calculate daily Donchian channel (20-period) for breakout signals
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_daily[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily price breaks above Donchian(20) high with volume confirmation → long
        # 2. Daily price breaks below Donchian(20) low with volume confirmation → short
        # 3. Volatility regime filter: weekly ATR > 1.5% of price (avoid extremely low volatility)
        # 4. Volume confirmation: volume > 1.4x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: Donchian breakout above high with volume
        if (close[i] > donchian_high[i] and            # Price above Donchian high
            volume_ratio[i] > 1.4 and                  # Volume confirmation
            atr_14_daily[i] > 0.015 * close[i]):       # Volatility regime filter
            signals[i] = 0.25
            
        # Short conditions: Donchian breakdown below low with volume
        elif (close[i] < donchian_low[i] and           # Price below Donchian low
              volume_ratio[i] > 1.4 and                # Volume confirmation
              atr_14_daily[i] > 0.015 * close[i]):     # Volatility regime filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_Volume_ATR_Regime_Filter"
timeframe = "1d"
leverage = 1.0