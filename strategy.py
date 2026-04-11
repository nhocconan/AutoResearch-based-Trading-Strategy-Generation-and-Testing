#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR trailing stop
# - Long: price breaks above 20-period Donchian high with volume > 1.5x 20-period average volume
# - Short: price breaks below 20-period Donchian low with volume > 1.5x 20-period average volume
# - Exit: trailing stop at 2.5 * ATR(20) from extreme price (highest high for longs, lowest low for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by capturing breakouts with volatility filter

name = "4h_donchian_breakout_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Pre-compute 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 20-period ATR for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Donchian levels
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper band with volume confirmation
        if close_price > upper_band and vol_confirm:
            enter_long = True
        
        # Short breakout: price breaks below lower band with volume confirmation
        if close_price < lower_band and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high for long position
            highest_high = max(highest_high, high_price)
            # Exit long if price drops below highest_high - 2.5 * ATR
            exit_long = close_price < (highest_high - 2.5 * atr[i])
        elif position == -1:
            # Update lowest low for short position
            lowest_low = min(lowest_low, low_price)
            # Exit short if price rises above lowest_low + 2.5 * ATR
            exit_short = close_price > (lowest_low + 2.5 * atr[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            highest_high = high_price
            lowest_low = 0.0  # Reset for long
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            lowest_low = low_price
            highest_high = 0.0  # Reset for short
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            highest_high = 0.0
            lowest_low = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            highest_high = 0.0
            lowest_low = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals