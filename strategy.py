#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and close > 1w EMA50 (uptrend) with volume > 1.5x average.
Short when price breaks below lower Donchian channel and close < 1w EMA50 (downtrend) with volume > 1.5x average.
Uses 1d timeframe to target 30-100 total trades over 4 years. Donchian levels provide clear structure.
1w EMA50 ensures alignment with weekly trend. Volume confirmation filters weak breakouts.
ATR-based stoploss reduces drawdown in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_donchian = high_ma_20[i]
        lower_donchian = low_ma_20[i]
        ema50_val = ema50_1w_aligned[i]
        vol_ma_val = vol_ma_20[i]
        atr_val = atr[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1w EMA50 (uptrend) AND volume confirmation
            if (price > upper_donchian and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND price < 1w EMA50 (downtrend) AND volume confirmation
            elif (price < lower_donchian and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions: stoploss or trend reversal
            exit_signal = False
            
            if position == 1:
                # Stoploss: price drops below entry - 2*ATR
                # Trend reversal: price breaks below lower Donchian OR price < 1w EMA50
                if price <= entry_price - 2.0 * atr_val or price < lower_donchian or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Stoploss: price rises above entry + 2*ATR
                # Trend reversal: price breaks above upper Donchian OR price > 1w EMA50
                if price >= entry_price + 2.0 * atr_val or price > upper_donchian or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Volume_ATR"
timeframe = "1d"
leverage = 1.0