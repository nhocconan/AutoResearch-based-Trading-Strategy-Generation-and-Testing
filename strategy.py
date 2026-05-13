#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA(close) measures buying strength; Bear Power = Low - EMA(close) measures selling pressure.
# Long when Bull Power > 0 (buying pressure) AND 1w EMA34 rising (uptrend) AND volume > 1.3x average.
# Short when Bear Power < 0 (selling pressure) AND 1w EMA34 falling (downtrend) AND volume > 1.3x average.
# Uses ATR(20) trailing stop (1.8x) for risk control. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Elder Ray isolates market power dynamics; weekly EMA filter ensures trading with the higher timeframe trend.
# Works in bull markets (buying pressure + uptrend) and bear markets (selling pressure + downtrend).

name = "6h_ElderRay_BullBearPower_1wEMA34_VolumeSpike_ATRStop_v1"
timeframe = "6h"
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
    
    # Calculate ATR(20) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA(close) for Elder Ray (using 13-period as standard)
    close_series = pd.Series(close)
    ema_close = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_close  # Buying strength
    bear_power = low - ema_close   # Selling pressure
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 6h timeframe (wait for weekly bar to close)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate EMA34 slope for trend direction (rising/falling)
    ema34_slope = np.zeros_like(ema34_1w_aligned)
    ema34_slope[1:] = ema34_1w_aligned[1:] - ema34_1w_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_slope[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (buying pressure) AND EMA34 rising (uptrend) AND volume > 1.3x average
            if (bull_power[i] > 0 and 
                ema34_slope[i] > 0 and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Bear Power < 0 (selling pressure) AND EMA34 falling (downtrend) AND volume > 1.3x average
            elif (bear_power[i] < 0 and 
                  ema34_slope[i] < 0 and 
                  volume[i] > 1.3 * avg_volume[i]):
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
            # EXIT LONG: trailing stop hit (1.8x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 1.8 * atr[i])
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
            # EXIT SHORT: trailing stop hit (1.8x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 1.8 * atr[i])
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