#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume spike confirmation.
Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) 
          AND 1w EMA50 rising AND 12h volume > 2.0x 20-period MA.
Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment)
          AND 1w EMA50 falling AND 12h volume > 2.0x 20-period MA.
Exit when Alligator alignment breaks (Teeth crosses Jaw) or 1w EMA50 reverses.
Williams Alligator catches trends early with smoothed SMAs, reducing whipsaw in ranging markets.
1w EMA50 filter ensures we only trade with the major weekly trend, avoiding counter-trend losses.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (SMAs with specific periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # All shifted forward by 8, 5, 3 bars respectively
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for rolled values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 50, 20) + 8  # Alligator max period + jaw shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 2.0x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        # Alligator alignment conditions
        bullish_alignment = (teeth[i] > lips[i]) and (lips[i] > jaw[i])
        bearish_alignment = (teeth[i] < lips[i]) and (lips[i] < jaw[i])
        
        if position == 0:
            # Long: Bullish Alligator alignment AND price > Jaw AND EMA50 rising AND volume filter
            if bullish_alignment and (price > jaw[i]) and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND price < Jaw AND EMA50 falling AND volume filter
            elif bearish_alignment and (price < jaw[i]) and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment breaks (Teeth <= Lips) OR EMA50 starts falling
                if (teeth[i] <= lips[i]) or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment breaks (Teeth >= Lips) OR EMA50 starts rising
                if (teeth[i] >= lips[i]) or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0