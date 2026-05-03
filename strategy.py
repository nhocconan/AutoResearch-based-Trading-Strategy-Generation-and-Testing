#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation, and ATR-based stoploss.
# Long: Close breaks above Donchian(20) high AND price > 12h EMA50 (uptrend) AND volume > 1.5x 20-period MA
# Short: Close breaks below Donchian(20) low AND price < 12h EMA50 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout OR trend reversal OR ATR stoploss (2x ATR from entry)
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels capture volatility breakouts; 12h EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment.

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
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian high AND uptrend AND volume spike
            if close_val > upper_channel and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Close breaks below Donchian low AND downtrend AND volume spike
            elif close_val < lower_channel and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: Opposite breakout OR trend turns down OR ATR stoploss hit
            if (close_val < lower_channel or 
                not is_uptrend or 
                close_val <= entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Opposite breakout OR trend turns up OR ATR stoploss hit
            if (close_val > upper_channel or 
                not is_downtrend or 
                close_val >= entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals