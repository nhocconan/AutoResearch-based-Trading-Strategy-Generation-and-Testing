#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12-hour Donchian channel breakout with 12-hour EMA50 trend filter and volume confirmation.
In uptrend (price > 12h EMA50), buy breakouts above 12h Donchian upper channel; in downtrend (price < 12h EMA50), sell breakdowns below 12h Donchian lower channel.
Volume must exceed 2.0x 20-period average to confirm breakout strength. Exit on trend reversal or 1.5x ATR stop.
Designed for 10-20 trades/year (40-80 total over 4 years) to minimize fee drag while capturing major trend moves.
Works in bull markets via upper channel breakouts and in bear markets via lower channel breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Donchian and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe (wait for 12h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation (volume spike > 2.0x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(80, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian upper + uptrend + volume spike
            if (price_close > donchian_high_val and 
                price_close > ema_trend and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian lower + downtrend + volume spike
            elif (price_close < donchian_low_val and 
                  price_close < ema_trend and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # ATR-based stoploss (1.5x ATR from entry)
            if position == 1:
                # Approximate entry price as the Donchian high breakout level
                entry_approx = donchian_high_aligned[i-1] if i > 0 else donchian_high_aligned[i]
                if price_close < entry_approx - 1.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Approximate entry price as the Donchian low breakdown level
                entry_approx = donchian_low_aligned[i-1] if i > 0 else donchian_low_aligned[i]
                if price_close > entry_approx + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0