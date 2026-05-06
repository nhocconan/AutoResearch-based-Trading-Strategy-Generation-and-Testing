#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction and 1h Donchian breakout for entry
# - Uses 4h Supertrend (ATR multiplier 3.0) for trend direction
# - Uses 1h Donchian channels (20-period) for precise entry timing
# - Enters long when price breaks above 1h Donchian upper band AND 4h Supertrend is bullish
# - Enters short when price breaks below 1h Donchian lower band AND 4h Supertrend is bearish
# - Exits when price returns to 1h Donchian middle or opposite band
# - Session filter: only trade 08-20 UTC to avoid low-liquidity hours
# - Position size: 0.20 (20% of capital)
# - Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency

name = "1h_4hSupertrend_1hDonchian_Breakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend
    tr = np.maximum(high_4h - low_4h, 
                    np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), 
                               np.absolute(np.roll(low_4h, 1) - low_4h)))
    tr[0] = high_4h[0] - low_4h[0]
    
    atr_period = 10
    atr = np.zeros_like(high_4h)
    atr[atr_period-1] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period, len(high_4h)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate Supertrend
    factor = 3.0
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    supertrend = np.ones_like(high_4h)
    for i in range(1, len(high_4h)):
        if close_4h[i] > upper_band[i-1]:
            supertrend[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            supertrend[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            if supertrend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if supertrend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
    # Align 4h Supertrend to 1h timeframe
    supertrend_1h = align_htf_to_ltf(prices, df_4h, supertrend)
    
    # Calculate 1h Donchian channels
    donchian_period = 20
    upper_donchian = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_donchian = (upper_donchian + lower_donchian) / 2
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(supertrend_1h[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(middle_donchian[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 1h Donchian upper AND 4h Supertrend bullish
            if close[i] > upper_donchian[i] and supertrend_1h[i] == 1:
                signals[i] = 0.20
                position = 1
            # Short: break below 1h Donchian lower AND 4h Supertrend bearish
            elif close[i] < lower_donchian[i] and supertrend_1h[i] == -1:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to middle OR breaks below lower band
            if close[i] < middle_donchian[i] or close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to middle OR breaks above upper band
            if close[i] > middle_donchian[i] or close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals