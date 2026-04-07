#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversals with 1d EMA200 trend filter and volume confirmation
# Camarilla levels calculated from previous day's OHLC. Fade at R3/S3 (mean reversion),
# breakout continuation at R4/S4 (trend following). Use 1d EMA200 to determine trend direction
# for breakout filtering. Volume > 1.5x 6h average confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions.

name = "6h_camarilla_1d_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * range_ * 1.1 / 2  # R4 equivalent
    camarilla_h4 = prev_close + 1.1 * range_ * 1.1 / 4  # R3
    camarilla_h3 = prev_close + 1.1 * range_ * 1.1 / 6  # R2
    camarilla_l3 = prev_close - 1.1 * range_ * 1.1 / 6  # S2
    camarilla_l4 = prev_close - 1.1 * range_ * 1.1 / 4  # S3
    camarilla_l5 = prev_close - 1.1 * range_ * 1.1 / 2  # S4
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # 1d EMA200 trend filter
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # 6h volume average for confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(h5_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(l5_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma_6h_aligned[i]) or
            np.isnan(atr[i])):
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
            # Exit: mean reversion at H3 or breakout failure below H4
            elif close[i] < h3_aligned[i] or (close[i] > h4_aligned[i] and close[i] < h3_aligned[i]):
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
            # Exit: mean reversion at L3 or breakout failure above L4
            elif close[i] > l3_aligned[i] or (close[i] < l4_aligned[i] and close[i] > l3_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Mean reversion fade at extreme levels (R4/S4) when counter-trend
            # Long fade at S4 in uptrend, short fade at R4 in downtrend
            # Breakout continuation at R3/S3 when with trend
            
            # Long entry conditions:
            # 1. Fade: price < S4 and above EMA200 (fade in uptrend)
            # 2. Breakout: price > R3 and above EMA200 with volume (breakout in uptrend)
            long_fade = (close[i] < l4_aligned[i] and close[i] > ema_200_aligned[i])
            long_breakout = (close[i] > h3_aligned[i] and close[i] > ema_200_aligned[i] and 
                           volume[i] > 1.5 * volume_ma_6h_aligned[i])
            
            # Short entry conditions:
            # 1. Fade: price > R4 and below EMA200 (fade in downtrend)
            # 2. Breakout: price < L3 and below EMA200 with volume (breakout in downtrend)
            short_fade = (close[i] > h4_aligned[i] and close[i] < ema_200_aligned[i])
            short_breakout = (close[i] < l3_aligned[i] and close[i] < ema_200_aligned[i] and 
                            volume[i] > 1.5 * volume_ma_6h_aligned[i])
            
            if long_fade or long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_fade or short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals