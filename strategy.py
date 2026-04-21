#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA trend + volume confirmation + ATR stoploss
# Long when price breaks above Donchian high(20), price > 12h EMA50, and 4h volume > 1.5x 20-bar average
# Short when price breaks below Donchian low(20), price < 12h EMA50, and 4h volume > 1.5x 20-bar average
# Exit when price crosses 12h EMA50 or ATR-based stoploss hit
# Donchian provides clear breakout levels, EMA filters trend direction, volume confirms conviction
# Target: 25-40 trades/year by requiring trend alignment + volume spike + breakout confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume moving average (20-period)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema50_val = ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirm = volume > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian high, price > EMA50, volume confirmation
            if price > donch_high_val and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low, price < EMA50, volume confirmation
            elif price < donch_low_val and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below EMA50 or ATR stoploss hit
                if price < ema50_val or price < entry_price - 2.0 * atr_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above EMA50 or ATR stoploss hit
                if price > ema50_val or price > entry_price + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0