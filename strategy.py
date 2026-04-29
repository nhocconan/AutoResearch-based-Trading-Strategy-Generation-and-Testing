#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR-based trailing stop
# Donchian breakouts capture momentum bursts after consolidation
# Daily EMA34 ensures we trade breakouts in alignment with higher timeframe trend
# Volume confirmation validates breakout strength
# ATR trailing stop manages risk and allows profits to run
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by daily EMA)
# Target: 20-50 trades/year (80-200 total over 4 years)

name = "4h_Donchian20_1dEMA34_Trend_Volume_ATRStop_v1"
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
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for 4h data for volatility and stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian Channel(20) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stop
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    start_idx = max(40, 34, 20, 20, 14)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available or ATR not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish breakout: price breaks above 20-period high in bullish daily regime
            if curr_close > highest_high[i] and curr_volume_confirm and curr_close > curr_ema34_1d:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Bearish breakout: price breaks below 20-period low in bearish daily regime
            elif curr_close < lowest_low[i] and curr_volume_confirm and curr_close < curr_ema34_1d:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry for trailing stop
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # ATR-based trailing stop: exit if price drops 2.5*ATR from highest since entry
            if curr_close < (highest_since_entry - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest price since entry for trailing stop
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # ATR-based trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if curr_close > (lowest_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals