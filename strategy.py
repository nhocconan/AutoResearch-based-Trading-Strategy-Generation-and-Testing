#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-week EMA trend and volume confirmation.
# Uses 1-week EMA20 to establish trend bias (long above EMA20, short below EMA20).
# Breakouts in direction of EMA trend with volume capture institutional moves.
# Designed for 4h timeframe to target 75-200 trades over 4 years with proven structure.
# Works in bull/bear markets via EMA-based directional bias and volume confirmation.

name = "4h_donchian20_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week EMA20 for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly closes
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 / 21) + (ema_20_1w[i-1] * 19 / 21)
    
    # Align EMA20 to 4h timeframe (shifted by 1 week for no look-ahead)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA20, short below EMA20
        bullish_bias = close[i] > ema_20_aligned[i]
        bearish_bias = close[i] < ema_20_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below EMA20 or stoploss (2x ATR approximation using Donchian width)
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] < ema_20_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above EMA20 or stoploss
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] > ema_20_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-week EMA trend and volume confirmation.
# Uses 1-week EMA20 to establish trend bias (long above EMA20, short below EMA20).
# Breakouts in direction of EMA trend with volume capture institutional moves.
# Designed for 4h timeframe to target 75-200 trades over 4 years with proven structure.
# Works in bull/bear markets via EMA-based directional bias and volume confirmation.

name = "4h_donchian20_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week EMA20 for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly closes
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 / 21) + (ema_20_1w[i-1] * 19 / 21)
    
    # Align EMA20 to 4h timeframe (shifted by 1 week for no look-ahead)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA20, short below EMA20
        bullish_bias = close[i] > ema_20_aligned[i]
        bearish_bias = close[i] < ema_20_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below EMA20 or stoploss (2x ATR approximation using Donchian width)
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] < ema_20_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above EMA20 or stoploss
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] > ema_20_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals