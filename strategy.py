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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # These levels act as intraday support/resistance
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price = (h_12h + l_12h + c_12h) / 3.0
    range_12h = h_12h - l_12h
    
    # Camarilla levels
    r3 = typical_price + range_12h * 1.1 / 4.0
    s3 = typical_price - range_12h * 1.1 / 4.0
    r4 = typical_price + range_12h * 1.1 / 2.0
    s4 = typical_price - range_12h * 1.1 / 2.0
    
    # Align HTF indicators to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    r4_6h = align_htf_to_ltf(prices, df_12h, r4)
    s4_6h = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate 6h ATR for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(atr[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic:
        # LONG: Price breaks above R3 with volume confirmation → target R4
        # SHORT: Price breaks below S3 with volume confirmation → target S4
        # Volatility filter: ATR > 0.3% of price
        # Volume confirmation: volume > 1.4x average
        # Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above R3
        if (close[i] > r3_6h[i] and            # Price above R3 Camarilla level
            volume_ratio[i] > 1.4 and          # Volume confirmation
            atr[i] > 0.003 * close[i]):        # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below S3
        elif (close[i] < s3_6h[i] and          # Price below S3 Camarilla level
              volume_ratio[i] > 1.4 and        # Volume confirmation
              atr[i] > 0.003 * close[i]):      # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0