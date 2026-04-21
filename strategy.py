#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1-week Donchian channel breakout with volume confirmation and ATR-based stop.
In uptrend (price > weekly SMA50), buy breakouts above weekly Donchian high; in downtrend (price < weekly SMA50),
sell breakdowns below weekly Donchian low. Volume must exceed 1.5x 20-period average to confirm breakout.
Designed for 10-30 trades/year (40-120 total over 4 years) to minimize fee drag while capturing directional moves.
Uses 1-week timeframe for trend and structure, 1d for execution to reduce whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for Donchian and SMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period weekly Donchian channels
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period weekly SMA for trend filter
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe (wait for weekly bar to close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(sma_50_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        sma_trend = sma_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high in uptrend
            if (price_high > upper and 
                price_close > sma_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low in downtrend
            elif (price_low < lower and 
                  price_close < sma_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < sma_trend:
                exit_signal = True
            elif position == -1 and price_close > sma_trend:
                exit_signal = True
            
            # ATR-based stoploss (2.5x ATR from breakout/breakdown level)
            if position == 1:
                entry_approx = upper  # Entered near weekly Donchian high
                if price_close < entry_approx - 2.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                entry_approx = lower  # Entered near weekly Donchian low
                if price_close > entry_approx + 2.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchian20_50SMA_Volume_ATR"
timeframe = "1d"
leverage = 1.0