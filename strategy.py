#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-week EMA trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > weekly EMA50 (uptrend) + volume > 1.5x average
# Short when Williams %R > -20 (overbought) + price < weekly EMA50 (downtrend) + volume > 1.5x average
# Exit when Williams %R crosses -50 (mean reversion) or volume drops below average
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-week EMA for trend and volume spike for confirmation
# Target: 80-160 total trades over 4 years (20-40/year)

name = "6h_williamsr_1w_ema_vol_v1"
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
    volume = prices['volume'].values
    
    # 1-week data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA(50)
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_50 = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(willr[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses -50 or volume drops below average
            elif willr[i] > -50 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses -50 or volume drops below average
            elif willr[i] < -50 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes with trend and volume confirmation
            # Volume filter: volume > 1.5x average (strong participation)
            vol_spike = volume[i] > 1.5 * vol_ma[i]
            
            # Long: Williams %R oversold + price above weekly EMA50 + volume spike
            if willr[i] < -80 and close[i] > ema_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R overbought + price below weekly EMA50 + volume spike
            elif willr[i] > -20 and close[i] < ema_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals