#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + Elder Ray + volume spike filter
    # Long when: Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND volume > 1.5x 20-bar avg
    # Short when: Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND volume > 1.5x 20-bar avg
    # Exit when: Alligator alignment reverses OR volume drops below average
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Alligator identifies trend direction and exhaustion; Elder Ray measures bull/bear power;
    # Volume spike confirms institutional participation. Works in bull (trend continuation) and bear (strong moves only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (requires daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (SMAs with specific periods)
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    def sma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            for i in range(period-1, len(data)):
                result[i] = np.mean(data[i-period+1:i+1])
        return result
    
    jaw = sma(close, 13)
    teeth = sma(close, 8)
    lips = sma(close, 5)
    
    # Shift Alligator lines (jaw: +8, teeth: +5, lips: +3)
    jaw_shifted = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth_shifted = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips_shifted = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Calculate Elder Ray (requires 1d data)
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    def ema(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            multiplier = 2 / (period + 1)
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema13 = ema(close_1d, 13)
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align HTF indicators to 12h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions (using previous bar's values to avoid look-ahead)
        bullish_alignment = (jaw_aligned[i-1] < teeth_aligned[i-1]) and (teeth_aligned[i-1] < lips_aligned[i-1])
        bearish_alignment = (jaw_aligned[i-1] > teeth_aligned[i-1]) and (teeth_aligned[i-1] > lips_aligned[i-1])
        
        # Elder Ray conditions
        strong_bull = bull_power_aligned[i-1] > 0
        strong_bear = bear_power_aligned[i-1] < 0
        
        # Entry conditions with volume confirmation
        long_entry = bullish_alignment and strong_bull and volume_confirmed[i-1] and position != 1
        short_entry = bearish_alignment and strong_bear and volume_confirmed[i-1] and position != -1
        
        # Exit conditions
        exit_long = position == 1 and (not bullish_alignment or not strong_bull or not volume_confirmed[i-1])
        exit_short = position == -1 and (not bearish_alignment or not strong_bear or not volume_confirmed[i-1])
        
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

name = "12h_1d_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0