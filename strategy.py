#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
# Williams Alligator: Jaw (EMA13, 8-shift), Teeth (EMA8, 5-shift), Lips (EMA5, 3-shift).
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 1.8x 20-period average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 1.8x 20-period average.
# Exit on ATR(14) trailing stop (2.0x). Uses 12h primary timeframe and 1d HTF for trend alignment.
# Alligator identifies trend formation and absence of trend. 1d EMA34 filters intermediate trend,
# volume spike confirms breakout authenticity. Designed for BTC/ETH with strict entry to avoid overtrading.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams Alligator components
    # Jaw: EMA(13) of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # Shift 8 bars forward
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: EMA(8) of median price, shifted 5 bars
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # Shift 5 bars forward
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: EMA(5) of median price, shifted 3 bars
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # Shift 3 bars forward
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Get 1d data for EMA34 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF arrays to 12h timeframe (wait for completed 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 12h volume > 1.8x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
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