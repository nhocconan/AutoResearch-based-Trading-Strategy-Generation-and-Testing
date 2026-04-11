#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR-based stoploss
# - Long: price breaks above Donchian upper (20-period high) + volume > 1.5x 20-period avg
# - Short: price breaks below Donchian lower (20-period low) + volume > 1.5x 20-period avg
# - Exit: ATR trailing stop (2.0 ATR from extreme price) or opposite Donchian break
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves effectively in both bull and bear markets
# - Volume confirmation filters out false breakouts
# - ATR stoploss manages risk during adverse moves

name = "4h_12h_donchian_volume_atr_v2"
timeframe = "4h"
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
    highest_since_entry = 0.0  # for long ATR trailing stop
    lowest_since_entry = 0.0   # for short ATR trailing stop
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h volume 20-period SMA
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above upper Donchian channel with volume confirmation
        if close_price > upper_channel and vol_confirm:
            enter_long = True
        
        # Short breakdown: price below lower Donchian channel with volume confirmation
        if close_price < lower_channel and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest price since entry for trailing stop
            highest_since_entry = max(highest_since_entry, close_price)
            # Exit long if ATR trailing stop hit or opposite Donchian break
            exit_long = (close_price <= highest_since_entry - 2.0 * atr_14[i]) or (close_price < lower_channel)
        elif position == -1:
            # Update lowest price since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, close_price)
            # Exit short if ATR trailing stop hit or opposite Donchian break
            exit_short = (close_price >= lowest_since_entry + 2.0 * atr_14[i]) or (close_price > upper_channel)
        
        # Track entry price for stoploss initialization
        if enter_long or enter_short:
            entry_price = close_price
            highest_since_entry = entry_price
            lowest_since_entry = entry_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals