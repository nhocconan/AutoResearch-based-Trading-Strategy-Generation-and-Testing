#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Uses daily Camarilla levels (based on previous day's range) to identify reversal zones
# Long at S1/S2 with 1d uptrend (close > EMA50) and volume > 1.5x 6h average
# Short at R1/R2 with 1d downtrend (close < EMA50) and volume > 1.5x 6h average
# Exit when price moves to opposite Camarilla level or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_camarilla_reversal_1d_ema_vol_v1"
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
    
    # 6h data for calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # 1d data for Camarilla calculation (uses previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (based on previous day)
    # Camarilla formulas: 
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # H2 = close + 0.55 * (high - low)
    # L2 = close - 0.55 * (high - low)
    # H1 = close + 0.275 * (high - low)
    # L1 = close - 0.275 * (high - low)
    # We'll use H3/L3 for reversals and H4/L4 for breakouts
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # handle first value
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    daily_range = prev_high - prev_low
    
    # Camarilla levels based on previous day
    H4 = prev_close + 1.5 * daily_range
    L4 = prev_close - 1.5 * daily_range
    H3 = prev_close + 1.1 * daily_range
    L3 = prev_close - 1.1 * daily_range
    H2 = prev_close + 0.55 * daily_range
    L2 = prev_close - 0.55 * daily_range
    H1 = prev_close + 0.275 * daily_range
    L1 = prev_close - 0.275 * daily_range
    
    # Align Camarilla levels to 6h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h volume average for confirmation
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_6h_aligned[i]) or 
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
            # Exit: price reaches H3 (opposite level) or trend reverses
            elif close[i] >= H3_aligned[i] or close[i] < ema_1d_aligned[i]:
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
            # Exit: price reaches L3 (opposite level) or trend reverses
            elif close[i] <= L3_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for reversals at Camarilla levels with volume confirmation
            # Long at L3: price touches/slightly breaks L3 with 1d uptrend and volume spike
            if (close[i] <= L3_aligned[i] * 1.002 and  # allow small penetration
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short at H3: price touches/slightly breaks H3 with 1d downtrend and volume spike
            elif (close[i] >= H3_aligned[i] * 0.998 and  # allow small penetration
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma_6h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals