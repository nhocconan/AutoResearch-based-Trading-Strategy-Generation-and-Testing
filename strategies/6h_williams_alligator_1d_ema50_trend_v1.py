#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day trend filter
# Long when price > Alligator's Jaw and Alligator is bullish (Teeth > Lips) and 1-day close > 1-day EMA50
# Short when price < Alligator's Jaw and Alligator is bearish (Teeth < Lips) and 1-day close < 1-day EMA50
# Exit when price crosses Alligator's Teeth (middle line)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Uses 1-day EMA50 for trend filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_williams_alligator_1d_ema50_trend_v1"
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
    
    # Williams Alligator (SMA-based)
    # Jaw: Blue line - 13-period SMA smoothed by 8 periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: Red line - 8-period SMA smoothed by 5 periods
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values
    
    # Lips: Green line - 5-period SMA smoothed by 3 periods
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: price crosses below Alligator's Teeth
            elif close[i] < teeth[i]:
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
            # Exit: price crosses above Alligator's Teeth
            elif close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment with 1-day trend filter
            # Bullish Alligator: Teeth > Lips (red above green)
            # Bearish Alligator: Teeth < Lips (red below green)
            bullish_alligator = teeth[i] > lips[i]
            bearish_alligator = teeth[i] < lips[i]
            
            # Long: price > Jaw AND bullish Alligator AND 1-day close > EMA50
            if close[i] > jaw[i] and bullish_alligator and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price < Jaw AND bearish Alligator AND 1-day close < EMA50
            elif close[i] < jaw[i] and bearish_alligator and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals