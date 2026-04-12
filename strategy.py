#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 34-period EMA filter (7-month EMA equivalent)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly 34-period EMA
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 20-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr20 = np.full(n, np.nan)
    for i in range(19, n):
        atr20[i] = np.nanmean(tr[i-19:i+1])
    
    # Calculate 50-period ATR EMA for volatility regime
    atr_ema50 = np.full(n, np.nan)
    atr_series = pd.Series(atr20)
    atr_ema50_values = atr_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ema50[:] = atr_ema50_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr20[i]) or 
            np.isnan(atr_ema50[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR20 > 1.2x 50-period ATR EMA (elevated volatility)
        vol_filter = atr20[i] > atr_ema50[i] * 1.2
        
        # Trend filter: price above/below weekly 34 EMA
        price_above_ema34 = close[i] > ema34_1w_aligned[i]
        price_below_ema34 = close[i] < ema34_1w_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high[i-1]  # break above previous high
        short_breakout = close[i] < low[i-1]  # break below previous low
        
        long_entry = long_breakout and price_above_ema34 and vol_filter
        short_entry = short_breakout and price_below_ema34 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema34_1w_aligned[i]) or (atr20[i] < atr_ema50[i] * 0.8)
        short_exit = (close[i] > ema34_1w_aligned[i]) or (atr20[i] < atr_ema50[i] * 0.8)
        
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

name = "1d_1w_ema34_breakout_vol_filter_v1"
timeframe = "1d"
leverage = 1.0