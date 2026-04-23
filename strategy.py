#!/usr/bin/env python3
"""
Hypothesis: 1d Bollinger Band squeeze breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper BB(20,2) AND price > 1w EMA50 AND volume > 1.5x average.
Short when price breaks below lower BB(20,2) AND price < 1w EMA50 AND volume > 1.5x average.
Exit when price crosses middle BB (20-period SMA) or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year per symbol.
Bollinger squeeze breakouts capture volatility expansion phases, effective in both trending and ranging markets.
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
    
    # Load 1d data for Bollinger Bands calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Bollinger Bands(20,2) on 1d data
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20  # 20-period SMA
    
    # Calculate ATR(14) on 1d data for stoploss
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or
            np.isnan(atr_1d[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 1d close for price comparison
        price_1d = close_1d[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB AND price > 1w EMA50 AND volume confirmation
            if (price_1d > upper_bb[i] and 
                price_1d > ema50_1w_aligned[i] and 
                volume_1d[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_1d
            # Short: price breaks below lower BB AND price < 1w EMA50 AND volume confirmation
            elif (price_1d < lower_bb[i] and 
                  price_1d < ema50_1w_aligned[i] and 
                  volume_1d[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_1d
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle BB (mean reversion)
                if price_1d < middle_bb[i]:
                    exit_signal = True
                # ATR-based stoploss
                elif price_1d < entry_price - 2.5 * atr_1d[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle BB (mean reversion)
                if price_1d > middle_bb[i]:
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

name = "1D_BollingerSqueeze_1wEMA50_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0