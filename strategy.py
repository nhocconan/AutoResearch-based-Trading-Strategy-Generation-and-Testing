#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR-based stoploss.
Long when price breaks above 20-period Donchian high AND volume > 1.5x 20-period average volume.
Short when price breaks below 20-period Donchian low AND volume > 1.5x 20-period average volume.
Exit when price crosses the 10-period EMA (mean reversion) or ATR stoploss hit.
Uses 1d HTF for trend filter (EMA50) to reduce whipsaws in ranging markets.
Target: 75-200 total trades over 4 years (19-50/year) for BTC/ETH/SOL.
Donchian channels provide clear structure, volume confirms conviction, EMA10 exit captures swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period average volume for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 10-period EMA for exit signal
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 10)  # Donchian (20), ATR (14), EMA10 (10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_10[i]) or np.isnan(atr[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma_val = vol_ma[i]
        ema10_val = ema_10[i]
        atr_val = atr[i]
        ema50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian high AND volume spike AND price > 1d EMA50 (bullish bias)
            if price > upper and volume[i] > 1.5 * vol_ma_val and price > ema50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Break below Donchian low AND volume spike AND price < 1d EMA50 (bearish bias)
            elif price < lower and volume[i] > 1.5 * vol_ma_val and price < ema50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Calculate stoploss levels
            if position == 1:
                stop_loss = entry_price - 2.5 * atr_val
                # Long exit: price < EMA10 OR price < stop loss
                if price < ema10_val or price < stop_loss:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                stop_loss = entry_price + 2.5 * atr_val
                # Short exit: price > EMA10 OR price > stop loss
                if price > ema10_val or price > stop_loss:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_Breakout_VolumeConfirmation_EMA10Exit"
timeframe = "4h"
leverage = 1.0