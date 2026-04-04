#!/usr/bin/env python3
"""
exp_6452_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume spike.
Works in bull/bear: Donchian breakout captures strong moves; EMA filter avoids counter-trend trades.
Target 50-150 total trades over 4 years (12-37/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6452_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = prices['volume'].rolling(window=20, min_periods=20).mean()
    volume_spike = prices['volume'] > (1.5 * vol_ma_20)
    
    # Donchian channels on primary timeframe
    donchian_high_20 = prices['high'].rolling(window=20, min_periods=20).max()
    donchian_low_20 = prices['low'].rolling(window=20, min_periods=20).min()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if no position size yet
        if i < 20:
            continue
            
        close = prices['close'].iloc[i]
        
        # Get aligned HTF values
        ema_50 = ema_50_1d_aligned[i]
        
        # Long conditions: price breaks above Donchian high + above 1d EMA + volume spike
        long_cond = (close > donchian_high_20.iloc[i] and 
                     close > ema_50 and 
                     volume_spike.iloc[i])
        
        # Short conditions: price breaks below Donchian low + below 1d EMA + volume spike
        short_cond = (close < donchian_low_20.iloc[i] and 
                      close < ema_50 and 
                      volume_spike.iloc[i])
        
        # Exit conditions: ATR-based stoploss or opposite signal
        if position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            atr = calculate_atr(prices, i, 14)
            if atr > 0 and close < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
            # Reverse signal
            elif short_cond:
                signals[i] = -0.30  # Short 30%
                position = -1
                entry_price = close
            else:
                signals[i] = 0.30  # Maintain long
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            atr = calculate_atr(prices, i, 14)
            if atr > 0 and close > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
            # Reverse signal
            elif long_cond:
                signals[i] = 0.30  # Long 30%
                position = 1
                entry_price = close
            else:
                signals[i] = -0.30  # Maintain short
        else:  # Flat
            if long_cond:
                signals[i] = 0.30  # Long 30%
                position = 1
                entry_price = close
            elif short_cond:
                signals[i] = -0.30  # Short 30%
                position = -1
                entry_price = close
            else:
                signals[i] = 0.0
    
    return signals

def calculate_atr(prices, idx, period):
    """Calculate ATR for given index"""
    if idx < period:
        return 0.0
    high = prices['high'].iloc[idx-period+1:idx+1]
    low = prices['low'].iloc[idx-period+1:idx+1]
    close = prices['close'].iloc[idx-period+1:idx+1]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean().iloc[-1]
    return atr if not pd.isna(atr) else 0.0