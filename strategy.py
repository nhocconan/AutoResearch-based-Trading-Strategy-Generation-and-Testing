#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 12-period EMA filter (12-day EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 12:
        return np.zeros(n)
    
    # Calculate daily 12-period EMA
    close_1d = df_1d['close'].values
    ema12_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_1d_aligned = align_htf_to_ltf(prices, df_1d, ema12_1d)
    
    # Calculate 24-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(23, n):
        atr[i] = np.nanmean(tr[i-23:i+1])
    
    # Calculate 50-period ATR SMA for volatility regime
    atr_sma50 = np.full(n, np.nan)
    atr_series = pd.Series(atr)
    atr_sma50_values = atr_series.rolling(window=50, min_periods=50).mean().values
    atr_sma50[:] = atr_sma50_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema12_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_sma50[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.0x 50-period ATR SMA (elevated volatility)
        vol_filter = atr[i] > atr_sma50[i] * 1.0
        
        # Trend filter: price above/below daily 12 EMA
        price_above_ema12 = close[i] > ema12_1d_aligned[i]
        price_below_ema12 = close[i] < ema12_1d_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high[i-1]  # break above previous high
        short_breakout = close[i] < low[i-1]  # break below previous low
        
        long_entry = long_breakout and price_above_ema12 and vol_filter
        short_entry = short_breakout and price_below_ema12 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema12_1d_aligned[i]) or (atr[i] < atr_sma50[i] * 0.8)
        short_exit = (close[i] > ema12_1d_aligned[i]) or (atr[i] < atr_sma50[i] * 0.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_ema12_breakout_vol_filter_v1"
timeframe = "12h"
leverage = 1.0