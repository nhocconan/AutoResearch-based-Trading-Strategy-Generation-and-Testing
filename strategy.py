#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation
# H4/L4 represent strong intraday support/resistance; breakouts with volume and 1d trend alignment
# capture sustained moves. EMA34 on 1d filters counter-trend breakouts in ranging markets.
# Works in bull/bear: in uptrend, longs from H4 breakouts; in downtrend, shorts from L4 breakdowns.
# Volume confirmation reduces false breakouts. Target 12-35 trades/year.

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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    camarilla_h4 = daily_close + 1.1 * (daily_high - daily_low) / 2.0
    camarilla_l4 = daily_close - 1.1 * (daily_high - daily_low) / 2.0
    
    # Align HTF Camarilla levels to 6h timeframe
    h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h4_6h[i]) or np.isnan(l4_6h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: 6h breakout above H4 with volume confirmation and 1d uptrend
        if (close[i] > h4_6h[i] and                    # 6h price above H4 camarilla level
            volume_ratio[i] > 1.5 and                  # Strong volume confirmation
            close[i] > ema_34_1d_aligned[i]):          # 1d uptrend filter
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below L4 with volume confirmation and 1d downtrend
        elif (close[i] < l4_6h[i] and                  # 6h price below L4 camarilla level
              volume_ratio[i] > 1.5 and                # Strong volume confirmation
              close[i] < ema_34_1d_aligned[i]):        # 1d downtrend filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_H4_L4_Breakout_Volume_EMA34_Filter"
timeframe = "6h"
leverage = 1.0