#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Choppiness Index regime filter + 12-hour RSI mean reversion
# In ranging markets (CHOP > 61.8), buy when RSI < 30, sell when RSI > 70
# In trending markets (CHOP < 38.2), buy when RSI > 50 and rising, sell when RSI < 50 and falling
# Uses 12h timeframe for regime and signal generation to avoid overtrading
# Position size: 0.25
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_chop_rsi_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12-hour data for regime and signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Choppiness Index (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (sum of TR over 14 periods)
    atr_12h = np.zeros_like(tr_12h)
    for i in range(len(tr_12h)):
        if i < 14:
            atr_12h[i] = np.nan
        else:
            atr_12h[i] = np.sum(tr_12h[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = np.zeros_like(high_12h)
    lowest_low_14 = np.zeros_like(low_12h)
    for i in range(len(high_12h)):
        if i < 13:
            highest_high_14[i] = np.nan
            lowest_low_14[i] = np.nan
        else:
            highest_high_14[i] = np.max(high_12h[i-13:i+1])
            lowest_low_14[i] = np.min(low_12h[i-13:i+1])
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = np.full_like(close_12h, np.nan)
    for i in range(13, len(close_12h)):
        if highest_high_14[i] > lowest_low_14[i] and atr_12h[i] > 0:
            chop[i] = 100 * np.log10(atr_12h[i] / (highest_high_14[i] - lowest_low_14[i])) / np.log10(14)
    
    # RSI (14-period) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # ATR for stoploss (6h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif chop_val > 61.8 and rsi_val > 70:  # overbought in range
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif chop_val < 38.2 and rsi_val < 50:  # weakening in trend
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif chop_val > 61.8 and rsi_val < 30:  # oversold in range
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif chop_val < 38.2 and rsi_val > 50:  # weakening in trend
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            if chop_val > 61.8:  # ranging market - mean reversion
                if rsi_val < 30:  # oversold
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif rsi_val > 70:  # overbought
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif chop_val < 38.2:  # trending market - momentum
                if rsi_val > 50:  # bullish momentum
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif rsi_val < 50:  # bearish momentum
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals