#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation.
# Long when price breaks above R1 AND 4h EMA20 rising AND volume > 2.0x 20-period average.
# Short when price breaks below S1 AND 4h EMA20 falling AND volume > 2.0x 20-period average.
# Uses ATR(14) trailing stop (2.0x) for risk control.
# Uses discrete position sizing (0.20) to minimize fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA20_VolumeConfirm_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from previous hour (for 1h timeframe)
    # R1 = close + 1.125*(high-low), S1 = close - 1.125*(high-low)
    # We use the previous hour's high, low, close for current hour's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar: use current values as fallback
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r1 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(20) on 4h data
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA20 to 1h timeframe (wait for 4h bar to close)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 errors)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade between 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # LONG: Price breaks above R1 AND 4h EMA20 rising AND volume spike AND in session
            if (close[i] > camarilla_r1[i] and ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] and 
                volume_confirm[i] and in_session):
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below S1 AND 4h EMA20 falling AND volume spike AND in session
            elif (close[i] < camarilla_s1[i] and ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] and 
                  volume_confirm[i] and in_session):
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