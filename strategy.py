#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Choppiness Index regime filter + 1d RSI mean reversion.
# In high chop (CHOP > 61.8): mean revert at 1d RSI extremes (long RSI<30, short RSI>70).
# In low chop (CHOP < 38.2): trend follow with 6s EMA(21) cross.
# Volume filter: require volume > 1.2x average to avoid low-liquidity whipsaws.
# Designed to work in both bull (trend follow) and bear (mean revert in ranges) markets.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6s_chop_rsi_regime_vol_v1"
timeframe = "6s"
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
    
    # 6s Choppiness Index (14-period)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_high_low = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_high_low / (atr * 14)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)  # avoid division by zero
    
    # 1d RSI(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6s EMA(21) for trend following
    ema_21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    # Volume filter: volume > 1.2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(chop[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_21[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit conditions
            if chop[i] > 61.8 and rsi_1d_aligned[i] > 50:  # chop high + RSI > 50
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] < ema_21[i]:   # trend regime + price < EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions
            if chop[i] > 61.8 and rsi_1d_aligned[i] < 50:  # chop high + RSI < 50
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] > ema_21[i]:   # trend regime + price > EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if volume_filter[i]:
                if chop[i] > 61.8:  # high chop: mean revert at RSI extremes
                    if rsi_1d_aligned[i] < 30:
                        signals[i] = 0.25
                        position = 1
                    elif rsi_1d_aligned[i] > 70:
                        signals[i] = -0.25
                        position = -1
                elif chop[i] < 38.2:  # low chop: trend follow with EMA cross
                    if close[i] > ema_21[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < ema_21[i]:
                        signals[i] = -0.25
                        position = -1
    
    return signals