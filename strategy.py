#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels with daily EMA trend filter and volume confirmation
# Uses daily Camarilla levels (H3/L3) for mean reversion entries, 1-week EMA for trend filter
# Requires volume > 1.8x 24-period average for confirmation (adapted for 12h timeframe)
# Implements 2.5x ATR(14) stoploss to manage risk
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Works in both bull and bear markets by fading extremes in ranging markets and following trend in trending markets

name = "12h_camarilla_1w_ema_vol_v1"
timeframe = "12h"
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
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3/L3, H4/L4
    # Range = previous day's high - low
    # H3 = close + 1.1 * range / 6
    # L3 = close - 1.1 * range / 6
    # H4 = close + 1.5 * range / 6
    # L4 = close - 1.5 * range / 6
    range_1d = prev_high - prev_low
    h3 = prev_close + 1.1 * range_1d / 6
    l3 = prev_close - 1.1 * range_1d / 6
    h4 = prev_close + 1.5 * range_1d / 6
    l4 = prev_close - 1.5 * range_1d / 6
    
    # Align daily Camarilla levels to 12h timeframe (2 bars per day)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1w EMA for trend filter (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume average (24-period for 12h = 12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
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
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches H3 or trend turns down
            elif close[i] >= h3_aligned[i] or close[i] < ema20_1w_aligned[i]:
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
            # Exit: price reaches L3 or trend turns up
            elif close[i] <= l3_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at extreme Camarilla levels
            # Long: price touches L4, in weekly uptrend, volume spike
            if (close[i] <= l4_aligned[i] and
                close[i] > ema20_1w_aligned[i] and
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches H4, in weekly downtrend, volume spike
            elif (close[i] >= h4_aligned[i] and
                  close[i] < ema20_1w_aligned[i] and
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals