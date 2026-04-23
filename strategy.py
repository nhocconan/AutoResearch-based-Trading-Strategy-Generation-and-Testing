#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA34 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1w EMA34 AND volume > 1.5x average.
Short when Williams %R > -20 (overbought) AND price < 1w EMA34 AND volume > 1.5x average.
Exit when Williams %R crosses above -50 for long or below -50 for short.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 10-25 trades/year per symbol.
Williams %R identifies overextended moves, effective in both trending and ranging markets when combined with trend filter.
"""

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
    
    # Load 1d data for Williams %R calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) on 1d data
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ATR(14) on 1d data for stoploss
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(atr_1d[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 1d close for price comparison
        price_1d = close_1d[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA34 AND volume confirmation
            if (williams_r[i] < -80 and 
                price_1d > ema34_1w_aligned[i] and 
                volume_1d[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_1d
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA34 AND volume confirmation
            elif (williams_r[i] > -20 and 
                  price_1d < ema34_1w_aligned[i] and 
                  volume_1d[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_1d
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (momentum weakening)
                if williams_r[i] > -50:
                    exit_signal = True
                # ATR-based stoploss
                elif price_1d < entry_price - 2.5 * atr_1d[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (momentum weakening)
                if williams_r[i] < -50:
                    exit_signal = True
                # ATR-based stoploss
                elif price_1d > entry_price + 2.5 * atr_1d[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsR_1wEMA34_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0