#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h/1d trend filter and volume confirmation.
# Long when price breaks above R1 AND 4h close > 4h EMA50 AND 1d EMA34 rising AND volume > 1.5x 20-bar avg.
# Short when price breaks below S1 AND 4h close < 4h EMA50 AND 1d EMA34 falling AND volume > 1.5x 20-bar avg.
# Uses ATR-based trailing stop (2.0x) for risk control. Discrete sizing 0.20.
# Camarilla pivot levels provide precise intraday support/resistance, effective in ranging and trending markets.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.

name = "1h_Camarilla_R1_S1_Breakout_4h1dTrend_Volume_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate Camarilla pivot levels (based on previous bar)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current vs 20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 AND 4h EMA50 uptrend AND 1d EMA34 rising AND volume spike
            if (close[i] > r1[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below S1 AND 4h EMA50 downtrend AND 1d EMA34 falling AND volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals