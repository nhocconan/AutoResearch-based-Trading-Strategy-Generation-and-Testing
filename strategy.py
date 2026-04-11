#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + volume confirmation + ATR stoploss
# - Long: price breaks above 20-day high with volume > 1.5x 20-day avg volume
# - Short: price breaks below 20-day low with volume > 1.5x 20-day avg volume
# - Exit: ATR-based trailing stop (3*ATR from extreme) or opposite Donchian breakout
# - Uses 1d timeframe for structure, avoids overtrading
# - Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# - Works in bull markets via breakouts, bear markets via short breakdowns

name = "1d_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Pre-compute 20-period indicators
    # Donchian channels: 20-period high/low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().values
    donchian_low = low_roll.min().values
    
    # Volume confirmation: 20-period average volume
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma[i]
        
        # Donchian levels
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # ATR for stoploss
        atr_value = atr[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper Donchian band with volume
        if close_price > upper_band and vol_confirm:
            enter_long = True
        
        # Short breakout: price breaks below lower Donchian band with volume
        if close_price < lower_band and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, high_price)
            # ATR trailing stop: exit if price drops 3*ATR from highest high
            if close_price < highest_high - 3.0 * atr_value:
                exit_long = True
            # Opposite breakout exit: exit long if price breaks below lower band
            elif close_price < lower_band:
                exit_long = True
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, low_price)
            # ATR trailing stop: exit if price rises 3*ATR from lowest low
            if close_price > lowest_low + 3.0 * atr_value:
                exit_short = True
            # Opposite breakout exit: exit short if price breaks above upper band
            elif close_price > upper_band:
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            highest_high = high_price
            lowest_low = low_price
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            highest_high = high_price
            lowest_low = low_price
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals