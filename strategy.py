#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ATR volatility filter.
    # Uses 12h Camarilla levels (H3/L3) as structure, entered only during high volatility + volume regimes.
    # Works in bull/bear by capturing explosive moves regardless of direction.
    # Target: 75-200 total trades over 4 years = 19-50/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels and ATR (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 4h data for volume MA (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous day's range)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla H3 and L3 levels
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume MA(20) for volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # Calculate 12h ATR MA(20) for volatility filter comparison
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ma_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_ma_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 12h ATR > 20-period mean (high volatility regime)
        volatility_filter = atr_aligned[i] > atr_ma_aligned[i]
        
        # Volume filter: current volume > 20-period MA (high volume regime)
        volume_filter = volume[i] > volume_ma_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = close[i] > camarilla_h3_aligned[i]  # Break above H3
        breakout_short = close[i] < camarilla_l3_aligned[i]  # Break below L3
        
        # Entry conditions: breakout with volatility AND volume filters
        long_entry = breakout_long and volatility_filter and volume_filter
        short_entry = breakout_short and volatility_filter and volume_filter
        
        # Exit conditions: price returns to opposite Camarilla level
        long_exit = close[i] < camarilla_l3_aligned[i]
        short_exit = close[i] > camarilla_h3_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_atr_volume_v1"
timeframe = "4h"
leverage = 1.0