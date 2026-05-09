#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with volume spike and weekly trend filter.
# Uses 4h timeframe with tight entry conditions to limit trades.
# R4/S4 breakouts capture strong momentum with 1.5x ATR confirmation.
# Weekly EMA200 filter ensures alignment with long-term trend for both bull and bear markets.
# Designed to avoid overtrading by requiring volume spike and trend alignment.
name = "4h_Camarilla_R4S4_Breakout_1wEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: Range = High - Low
    range_1d = high_1d - low_1d
    r4 = close_1d + (range_1d * 1.5000)
    s4 = close_1d - (range_1d * 1.5000)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r4_shifted = np.roll(r4, 1)
    s4_shifted = np.roll(s4, 1)
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # Weekly EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # ATR(14) for volatility confirmation
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike filter: volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema_200_4h[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume spike, above weekly EMA200, and > ATR(14) confirmation
            if (price > r4_4h[i] and vol_spike[i] and price > ema_200_4h[i] and 
                (price - r4_4h[i]) > (0.5 * atr[i])):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume spike, below weekly EMA200, and < -ATR(14) confirmation
            elif (price < s4_4h[i] and vol_spike[i] and price < ema_200_4h[i] and 
                  (s4_4h[i] - price) > (0.5 * atr[i])):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S4 (mean reversion to support)
            if price < s4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R4 (mean reversion to resistance)
            if price > r4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals