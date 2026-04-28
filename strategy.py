# Your task is to write a new strategy.py that follows the rules and builds upon the insights from the experiment history.
# Hypothesis: A 6h strategy using a 12h Supertrend filter and 12h Donchian breakout with volume confirmation.
# This combines a proven trend filter (Supertrend) with a classic breakout (Donchian) on a higher timeframe (12h) for direction,
# and uses volume on the 6h chart to confirm the breakout's strength.
# The Supertrend helps avoid whipsaws in ranging markets, while the Donchian breakout captures momentum.
# Using a 12h Supertrend on a 6h chart should provide a good balance of signal frequency and reliability.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Increased warmup for Supertrend
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend and Donchian
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Supertrend calculation (ATR=10, multiplier=3.0)
    # True Range
    tr1 = pd.Series(df_12h['high']).diff()
    tr2 = pd.Series(df_12h['low']).diff().abs()
    tr3 = abs(pd.Series(df_12h['high']) - pd.Series(df_12h['low']).shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)
    atr = tr.ewm(alpha=1/10, adjust=False, min_periods=10).mean()  # 10-period ATR
    
    # Basic Upper and Lower Bands
    hl2 = (pd.Series(df_12h['high']) + pd.Series(df_12h['low'])) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Final Upper and Lower Bands
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    for i in range(1, len(df_12h)):
        if df_12h['close'].iloc[i] <= final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]
        if df_12h['close'].iloc[i] >= final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]
    
    # Supertrend direction
    supertrend = np.zeros(len(df_12h))
    for i in range(1, len(df_12h)):
        if df_12h['close'].iloc[i] > final_upper.iloc[i-1]:
            supertrend[i] = 1
        elif df_12h['close'].iloc[i] < final_lower.iloc[i-1]:
            supertrend[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
    
    # 12h Donchian Channel (20)
    high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align 12h indicators to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: 12h Supertrend direction
        trend_up = supertrend_aligned[i] == 1
        trend_down = supertrend_aligned[i] == -1
        
        # Entry conditions: 
        # Long: breakout above 12h Donchian high in uptrend
        # Short: breakdown below 12h Donchian low in downtrend
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: opposite Donchian level touch
        long_exit = (close[i] < low_20_aligned[i]) and position == 1
        short_exit = (close[i] > high_20_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Supertrend_Donchian_Volume_Filter"
timeframe = "6h"
leverage = 1.0