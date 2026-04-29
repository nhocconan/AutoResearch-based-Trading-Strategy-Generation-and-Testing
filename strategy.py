#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation
# Long when Bull Power > 0 AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when Bear Power < 0 AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price crosses 12h EMA50 (trend change)
# Uses discrete position sizing (0.25) to reduce fee drag.
# Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to avoid overtrading.
# Elder Ray measures bull/bear strength relative to EMA13, providing clearer trend signals than price alone.
# Works in bull markets via strong Bull Power and in bear markets via strong Bear Power with trend alignment.

name = "6h_ElderRay_BullBearPower_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 12h data for EMA13 (used in Elder Ray calculation)
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_12h_aligned
    bear_power = low - ema_13_12h_aligned
    
    # Volume confirmation: >1.8x 20-bar average volume (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 12h EMA50 (trend change)
            if curr_close < curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 12h EMA50 (trend change)
            if curr_close > curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND price > 12h EMA50 AND volume confirmation
            if curr_bull_power > 0 and curr_close > curr_ema50_12h and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 AND price < 12h EMA50 AND volume confirmation
            elif curr_bear_power < 0 and curr_close < curr_ema50_12h and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals