#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour timeframe using Donchian(20) breakout with 1-day trend filter (EMA50),
# volume confirmation (>1.5x 20-period average), and ATR(14) stoploss.
# Long when price breaks above Donchian upper band AND price > EMA50 (1d) AND volume spike.
# Short when price breaks below Donchian lower band AND price < EMA50 (1d) AND volume spike.
# Exit when price crosses opposite Donchian band or ATR-based stoploss hits.
# Designed for fewer trades (<30/year) with strong trend capture in both bull and bear markets.
name = "12h_Donchian_EMA50_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1-day EMA50 for trend filter (updated only after daily close)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_multiplier = 2.5  # ATR multiplier for stoploss
    
    start_idx = max(20, 14, 50)  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        upper_band = high_roll[i]
        lower_band = low_roll[i]
        trend = ema50_1d_aligned[i]
        atr_val = atr[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above upper band AND price > trend AND volume spike
            if price > upper_band and price > trend and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below lower band AND price < trend AND volume spike
            elif price < lower_band and price < trend and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            # Exit if price crosses below lower band
            if price < lower_band:
                exit_signal = True
            # Exit if ATR-based stoploss hit (price < entry_price - atr_multiplier * atr)
            elif price < entry_price - atr_multiplier * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            # Exit if price crosses above upper band
            if price > upper_band:
                exit_signal = True
            # Exit if ATR-based stoploss hit (price > entry_price + atr_multiplier * atr)
            elif price > entry_price + atr_multiplier * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals