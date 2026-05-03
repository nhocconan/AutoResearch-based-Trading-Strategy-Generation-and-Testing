#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation, and ATR stoploss.
# Long: Close > Donchian Upper(20) AND Close > 12h EMA50 AND Volume > 1.5x 20-period MA
# Short: Close < Donchian Lower(20) AND Close < 12h EMA50 AND Volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or ATR-based stoploss (signal → 0 when price moves against position by 2.0*ATR)
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide structural breakouts; 12h EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via longs with trend alignment
# and in bear via shorts with trend alignment.

name = "4h_Donchian20_12hEMA50_Volume_ATR"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Update trailing stop for existing positions
        if position == 1:
            # Long stoploss: price < highest high since entry - 2.0 * ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short stoploss: price > lowest low since entry + 2.0 * ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        # Entry logic (only when flat)
        if position == 0:
            # Long: Close > Donchian Upper AND uptrend AND volume spike
            if close_val > upper and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Close < Donchian Lower AND downtrend AND volume spike
            elif close_val < lower and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
    
    return signals