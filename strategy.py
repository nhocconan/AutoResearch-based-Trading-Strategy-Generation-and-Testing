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
    
    # Get daily HTF data once before loop (4h primary, 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volatility filter and stoploss
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe with proper delay
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price above/below daily EMA50
        # 2. 4h Donchian breakout: price breaks above/below 20-period channel
        # 3. 4h volume confirmation: volume > 2.0x average
        # 4. ATR-based stoploss: exit when price moves against position by 2.5x ATR
        # 5. Discrete position sizing: 0.25
        
        # Track position for stoploss
        if i == 100:
            position = 0
            entry_price = 0
        
        # Update position based on signal
        if signals[i-1] != 0:
            position = np.sign(signals[i-1])
            entry_price = close[i-1]  # Approximate entry at previous bar's close
        
        # Check stoploss
        if position == 1 and close[i] < entry_price - 2.5 * atr_14_4h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > entry_price + 2.5 * atr_14_4h[i]:
            signals[i] = 0.0
            position = 0
        elif position == 0:
            # Look for new entries only when flat
            # Long conditions: break above Donchian high in uptrend
            if (close[i] > ema_50_4h[i] and          # Daily uptrend filter
                close[i] > highest_20[i] and         # Donchian breakout
                volume_ratio[i] > 2.0):              # Volume confirmation
                signals[i] = 0.25
                
            # Short conditions: break below Donchian low in downtrend
            elif (close[i] < ema_50_4h[i] and        # Daily downtrend filter
                  close[i] < lowest_20[i] and        # Donchian breakdown
                  volume_ratio[i] > 2.0):            # Volume confirmation
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_EMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0