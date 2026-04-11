#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d RSI filter.
# Long: price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + 1d RSI(14) > 50
# Short: price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + 1d RSI(14) < 50
# Exit: opposite Donchian breakout
# Uses volume and RSI on 1d to filter breakouts, ensuring institutional participation.
# Designed for 20-50 trades/year on 4h timeframe with focus on avoiding false breakouts.

name = "4h_1d_donchian_volume_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume moving average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    
    # Align 1d indicators to 4h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: 1d volume > 1.5 * 20-period average volume
        vol_filter = vol_ma_20_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # RSI filter: 1d RSI > 50 for long, < 50 for short
        rsi_filter_long = rsi_14_1d_aligned[i] > 50
        rsi_filter_short = rsi_14_1d_aligned[i] < 50
        
        # Entry conditions
        bullish_breakout = close[i] > donchian_high[i]
        bearish_breakout = close[i] < donchian_low[i]
        
        bullish_entry = bullish_breakout and vol_filter and rsi_filter_long
        bearish_entry = bearish_breakout and vol_filter and rsi_filter_short
        
        # Exit conditions: opposite breakout
        exit_long = bearish_breakout
        exit_short = bullish_breakout
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals