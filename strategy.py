#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR trailing stop
# - Long: Close breaks above 20-day Donchian high + 1w volume > 1.5x 20-period average
# - Short: Close breaks below 20-day Donchian low + 1w volume > 1.5x 20-period average
# - Exit: ATR(14) trailing stop (2.0 ATR from highest high/lowest low since entry)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - Volume confirmation filters false breakouts
# - ATR trailing stop allows trends to run while controlling risk

name = "1d_1w_donchian_volume_atr_v1"
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
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Load 1w data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w volume 20-period SMA
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for trailing stop (1d timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above Donchian high with volume confirmation
        if close_price > donchian_high and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below Donchian low with volume confirmation
        if close_price < donchian_low and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high since entry
            if high[i] > highest_high_since_entry:
                highest_high_since_entry = high[i]
            # Exit if price drops 2.0*ATR from highest high
            exit_long = close_price <= highest_high_since_entry - 2.0 * atr_14[i]
        elif position == -1:
            # Update lowest low since entry
            if low[i] < lowest_low_since_entry:
                lowest_low_since_entry = low[i]
            # Exit if price rises 2.0*ATR from lowest low
            exit_short = close_price >= lowest_low_since_entry + 2.0 * atr_14[i]
        
        # Track entry price and extreme levels for stoploss
        if enter_long or enter_short:
            entry_price = close_price
            highest_high_since_entry = high[i]
            lowest_low_since_entry = low[i]
        
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