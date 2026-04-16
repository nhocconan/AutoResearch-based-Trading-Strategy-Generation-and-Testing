#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extreme levels with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND price > EMA50 AND volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND price < EMA50 AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses -50 (mean reversion) or ATR-based stoploss (2.5x ATR).
# Uses discrete position size 0.25. 1d filters provide signal direction, 12h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Williams %R (14) ===
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # === 1d Indicators: EMA (50) ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1d Indicators: ATR (14) for stoploss ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        williams_r_val = williams_r_aligned[i]
        ema_50 = ema_50_aligned[i]
        atr = atr_14_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 12h volume average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit 1: Williams %R crosses above -50 (mean reversion)
            if williams_r_val > -50:
                exit_signal = True
            # Exit 2: ATR-based stoploss (2.5x ATR below entry)
            elif price < entry_price - 2.5 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit 1: Williams %R crosses below -50 (mean reversion)
            if williams_r_val < -50:
                exit_signal = True
            # Exit 2: ATR-based stoploss (2.5x ATR above entry)
            elif price > entry_price + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > EMA50 AND volume > 1.5x 20-period avg
            if (williams_r_val < -80) and (price > ema_50) and (vol > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) AND price < EMA50 AND volume > 1.5x 20-period avg
            elif (williams_r_val > -20) and (price < ema_50) and (vol > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dWilliamsR_EMA50_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0