#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
# Long when price breaks above 1h Camarilla R3 AND close > 4h EMA34 AND volume > 1.5x 20-period average.
# Short when price breaks below 1h Camarilla S3 AND close < 4h EMA34 AND volume > 1.5x 20-period average.
# Uses ATR-based trailing stop (2.0x) for risk control.
# Camarilla pivots provide intraday support/resistance, 4h EMA34 filters intermediate trend, volume spike confirms participation.
# Target: 15-37 trades/year (60-150 total over 4 years) on 1h timeframe with session filter (08-20 UTC).

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Volume_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1h data for Camarilla pivots (using previous bar's OHLC)
    # Camarilla levels based on previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar: use current close as previous
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels for 1h timeframe
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_upper = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_lower = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 on 4h close
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_upper[i]) or np.isnan(camarilla_lower[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade between 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            # Carry forward tracking values when filtered out
            if i > 0:
                if position == 1:
                    highest_since_entry[i] = highest_since_entry[i-1]
                elif position == -1:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price > 1h Camarilla R3 AND close > 4h EMA34 AND volume confirmation
            if close[i] > camarilla_upper[i] and close[i] > ema34_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < 1h Camarilla S3 AND close < 4h EMA34 AND volume confirmation
            elif close[i] < camarilla_lower[i] and close[i] < ema34_4h_aligned[i] and volume_confirm[i]:
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