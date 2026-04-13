#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams Alligator + 1w EMA(21) trend filter + volume confirmation
    # Long when: Alligator jaws < teeth < lips (bullish alignment) AND price > 1w EMA21 AND volume > 1.5x 50-bar avg volume
    # Short when: Alligator jaws > teeth > lips (bearish alignment) AND price < 1w EMA21 AND volume > 1.5x 50-bar avg volume
    # Exit when: Alligator alignment reverses OR price crosses 1w EMA21
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Alligator identifies trend via smoothed SMAs; 1w EMA21 filters higher timeframe trend;
    # Volume confirmation reduces false breakouts. Works in bull (trend continuation) and bear (mean-reversion at alignment).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams Alligator: 3 smoothed SMAs
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    # SMMA = smoothed moving average (EMA with alpha=1/period)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_6h, 13)  # 13-period SMMA
    teeth = smma(close_6h, 8)  # 8-period SMMA
    lips = smma(close_6h, 5)   # 5-period SMMA
    
    # Shift as per Alligator definition: Jaw shifted 8, Teeth 5, Lips 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Get 1w data for EMA(21) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips_shifted)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate volume confirmation: volume > 1.5x 50-bar average volume
    avg_volume = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions (using previous bar's values to avoid look-ahead)
        bullish_alignment = (jaw_aligned[i-1] < teeth_aligned[i-1]) and (teeth_aligned[i-1] < lips_aligned[i-1])
        bearish_alignment = (jaw_aligned[i-1] > teeth_aligned[i-1]) and (teeth_aligned[i-1] > lips_aligned[i-1])
        
        # 1w EMA21 trend filter
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = bullish_alignment and uptrend and volume_confirmed[i] and position != 1
        short_entry = bearish_alignment and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (not bullish_alignment or not uptrend))
        exit_short = (position == -1 and (not bearish_alignment or not downtrend))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_alligator_ema_volume_v1"
timeframe = "6h"
leverage = 1.0