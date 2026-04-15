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
    
    # Get weekly HTF data once before loop
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
    atr_14_w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly Donchian channels (20-period) based on previous week
    donchian_high_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align HTF Donchian levels to daily timeframe
    donchian_high_20_d = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_d = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate daily ATR(14) for stoploss and volatility filter
    tr1_d = high - low
    tr2_d = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_d = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_14_d = pd.Series(tr_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_d[i]) or np.isnan(donchian_low_20_d[i]) or 
            np.isnan(atr_14_d[i]) or np.isnan(volume_ratio[i]) or np.isnan(atr_14_w[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when weekly ATR > 1.5% of price (avoid low volatility sideways)
        volatility_regime = atr_14_w[i] > 0.015 * close[i]
        
        # Long conditions: daily breakout above weekly Donchian high with volume confirmation
        if (close[i] > donchian_high_20_d[i] and 
            volume_ratio[i] > 1.4 and 
            volatility_regime):
            signals[i] = 0.25
            
        # Short conditions: daily breakdown below weekly Donchian low with volume confirmation
        elif (close[i] < donchian_low_20_d[i] and 
              volume_ratio[i] > 1.4 and 
              volatility_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_ATR_Regime_Filter"
timeframe = "1d"
leverage = 1.0