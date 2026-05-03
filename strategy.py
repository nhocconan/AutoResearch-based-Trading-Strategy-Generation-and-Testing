#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.5x 20-period MA.
# Exit on opposite Donchian breakout or trend reversal (price crosses 1d EMA50).
# Uses ATR-based stoploss (signal=0 when price moves 2*ATR against position).
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.30.

name = "4h_Donchian20_1dEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Stoploss: 2*ATR against position
        if position == 1 and close_val < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close_val > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Donchian high AND above 1d EMA50 AND volume spike
            if close_val > donchian_high[i] and close_val > ema_50_1d_aligned[i] and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            # Short: price breaks below Donchian low AND below 1d EMA50 AND volume spike
            elif close_val < donchian_low[i] and close_val < ema_50_1d_aligned[i] and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below Donchian low OR below 1d EMA50
            if close_val < donchian_low[i] or close_val < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above Donchian high OR above 1d EMA50
            if close_val > donchian_high[i] or close_val > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals