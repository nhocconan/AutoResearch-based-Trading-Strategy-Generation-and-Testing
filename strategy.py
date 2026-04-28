#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA(50) trend filter and volume spike
# Camarilla pivot levels provide high-probability reversal/continuation points. Breakouts above R1 or below S1
# indicate momentum. 4h EMA ensures alignment with short-medium term trend. Volume spike confirms conviction.
# Designed for 1h timeframe targeting 15-37 trades/year to minimize fee drag while capturing strong moves.
# Works in both bull and bear markets by following trend direction via EMA filter.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 4h EMA to 1h (changes only when 4h bar closes)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: R1 = close + 1.091*(high-low), S1 = close - 1.091*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + 1.091 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 1.091 * (high_1d - low_1d)
    
    # Align Camarilla levels to 1h (using prior day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50, 14)  # volume MA(20), 4h EMA(50), ATR(14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Camarilla R1, above 4h EMA50, volume spike
            if price > camarilla_r1_aligned[i] and price > ema_50_4h_aligned[i] and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short entry: Price < Camarilla S1, below 4h EMA50, volume spike
            elif price < camarilla_s1_aligned[i] and price < ema_50_4h_aligned[i] and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or retracement to 4h EMA50
            # ATR-based stoploss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * atr[i]
            if price < stop_loss or price < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on stoploss or retracement to 4h EMA50
            # ATR-based stoploss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * atr[i]
            if price > stop_loss or price > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals