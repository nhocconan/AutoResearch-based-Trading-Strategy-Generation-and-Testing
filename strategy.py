#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) with 1w EMA50 uptrend and volume > 2x avg
# Short when price breaks below lower BB(20,2) with 1w EMA50 downtrend and volume > 2x avg
# Exit when price reverts to middle BB or trend changes
# Target: 30-100 total trades over 4 years with low frequency and high win rate
# ATR-based stoploss (2x ATR) to limit drawdown

name = "1d_bb_breakout_1w_ema50_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # EMA50 calculation on 1w
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    
    # Deviation
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    
    # Upper and Lower bands
    upper_band = basis + dev
    lower_band = basis - dev
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(bb_length, n):  # Start after BB warmup
        # Skip if required data not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(basis[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using high-low range
            atr_approx = high[i] - low[i]
            if close[i] < entry_price - 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below middle band or trend changes
            elif close[i] < basis[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            atr_approx = high[i] - low[i]
            if close[i] > entry_price + 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above middle band or trend changes
            elif close[i] > basis[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price breaks above upper BB, uptrend, high volume
            if (close[i] > upper_band[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB, downtrend, high volume
            elif (close[i] < lower_band[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals