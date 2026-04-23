#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 12h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 12h EMA50 AND volume > 1.8x 20-period average.
Exit when price crosses Donchian middle band (mean reversion) or ATR-based stoploss (2.0x ATR).
Uses discrete position sizing (0.30) to balance return and drawdown. Targets 20-50 trades/year per symbol.
Donchian channels provide structure, 12h EMA50 ensures trend alignment, volume filters weak breakouts.
Works in trending markets and avoids counter-trend trades. ATR stoploss manages risk during reversals.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_band = (highest_high + lowest_low) / 2
    
    # ATR(14) on 4h data for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 4h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        middle = middle_band[i]
        ema_trend = ema50_12h_aligned[i]
        
        if position == 0:
            # Long: Break above upper band + trend filter + volume spike
            if (price > upper_band and 
                price > ema_trend and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: Break below lower band + trend filter + volume spike
            elif (price < lower_band and 
                  price < ema_trend and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below middle band or ATR stoploss
                if price < middle:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above middle band or ATR stoploss
                if price > middle:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0