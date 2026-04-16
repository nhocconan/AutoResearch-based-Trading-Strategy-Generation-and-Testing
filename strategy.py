#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND price > 12h EMA34 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) lower band AND price < 12h EMA34 AND volume > 1.5x 20-period average.
# Exit when price returns to Donchian midpoint or ATR(10) < ATR(30) (contracting volatility).
# Uses discrete position size 0.25. Donchian provides clear breakout levels, 12h EMA ensures we trade with the higher timeframe trend,
# and volume confirmation reduces false signals. Target: 100-180 total trades over 4 years (25-45/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Middle band = (upper + lower) / 2
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(middle_20[i]) or np.isnan(atr_10[i]) or np.isnan(atr_30[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema_val = ema_34_12h_aligned[i]
        upper = upper_20[i]
        lower = lower_20[i]
        middle = middle_20[i]
        price = close[i]
        vol = volume[i]
        atr10 = atr_10[i]
        atr30 = atr_30[i]
        
        # Calculate 20-period volume average
        if i >= 20:
            vol_ma_20 = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma_20 = 0.0
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Volatility filter: ATR(10) > ATR(30) (expanding volatility)
        vol_filter_expanding = atr10 > atr30
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to midpoint or volatility contracts
            if price <= middle or not vol_filter_expanding:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to midpoint or volatility contracts
            if price >= middle or not vol_filter_expanding:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper band with trend and volume confirmation
            if price > upper and price > ema_val and vol_filter and vol_filter_expanding:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower band with trend and volume confirmation
            elif price < lower and price < ema_val and vol_filter and vol_filter_expanding:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0