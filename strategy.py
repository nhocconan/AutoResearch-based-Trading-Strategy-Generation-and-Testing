#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 AND 1d EMA34 rising AND volume > 1.5x 20-period MA.
# Short when price breaks below S3 AND 1d EMA34 falling AND volume > 1.5x 20-period MA.
# Uses ATR(14) trailing stop (2.0x) for risk control. Discrete sizing 0.25.
# Camarilla pivot levels provide strong intraday support/resistance that work in ranging and trending markets.
# 1d EMA34 filter ensures we only trade with the higher timeframe trend.
# Volume confirmation reduces false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume_ATRStop_v1"
timeframe = "12h"
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 1d (using previous day's OHLC)
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    #          S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # We use R3 and S3 as breakout levels
    rng = high_1d - low_1d
    r3_1d = close_1d + (rng * 1.1 / 4)
    s3_1d = close_1d - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above R3 AND 1d EMA34 rising AND volume confirmation
            if close[i] > r3_1d_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below S3 AND 1d EMA34 falling AND volume confirmation
            elif close[i] < s3_1d_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
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
                signals[i] = 0.25
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
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals