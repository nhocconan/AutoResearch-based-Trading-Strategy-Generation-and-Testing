#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike filter and ATR-based stoploss
# - Long: price breaks above Donchian upper band (20-period high) + volume > 2.0x 20-period average
# - Short: price breaks below Donchian lower band (20-period low) + volume > 2.0x 20-period average
# - Exit: ATR trailing stop (2.0 * ATR) from highest high/lowest low since entry
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves; volume filter ensures conviction
# - ATR stoploss adapts to volatility, reducing whipsaw in ranging markets

name = "4h_donchian_volume_atr_v1"
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
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Pre-compute Donchian channels (20-period) on 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for stoploss (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation (20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(atr_14[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above upper band + volume confirmation
        if close_price > upper_band and vol_confirm:
            enter_long = True
        
        # Short breakout: price below lower band + volume confirmation
        if close_price < lower_band and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high since entry
            if i == 0 or position != 1:  # Reset on new position
                highest_high_since_entry = close_price
            else:
                highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit long if price drops 2.0 * ATR from highest high
            exit_long = close_price <= (highest_high_since_entry - 2.0 * atr_14[i])
        
        elif position == -1:
            # Update lowest low since entry
            if i == 0 or position != -1:  # Reset on new position
                lowest_low_since_entry = close_price
            else:
                lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit short if price rises 2.0 * ATR from lowest low
            exit_short = close_price >= (lowest_low_since_entry + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation (optional, not used in ATR trail)
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            highest_high_since_entry = close_price
            lowest_low_since_entry = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            highest_high_since_entry = close_price
            lowest_low_since_entry = close_price
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