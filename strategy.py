#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Power (Bull/Bear) with 1-week volume confirmation.
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to detect trend strength.
# Volume confirmation ensures institutional participation in the move.
# Designed for 6h timeframe to target 50-150 trades over 4 years with low frequency.
# Works in both bull and bear markets by adapting to trend direction via EMA.

name = "6h_elderray1w_vol_v1"
timeframe = "6h"
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
    
    # 1-week EMA(13) for Elder Ray calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA calculation
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 13:
        ema_1w[12] = np.mean(close_1w[:13])
        for i in range(13, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (13 + 1)) + (ema_1w[i-1] * (11 / (13 + 1)))
    
    # Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = np.full(len(close_1w), np.nan)
    bear_power = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if not np.isnan(ema_1w[i]):
            bull_power[i] = df_1w['high'].iloc[i] - ema_1w[i]
            bear_power[i] = df_1w['low'].iloc[i] - ema_1w[i]
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    if len(vol_1w) >= 5:
        for i in range(4, len(vol_1w)):
            vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(13, 4)  # EMA needs 13, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear power turns positive (trend weakening) or stoploss
            if (bear_power_aligned[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull power turns negative (trend weakening) or stoploss
            if (bull_power_aligned[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: Bull power positive and increasing
                if (bull_power_aligned[i] > 0 and 
                    i > start and bull_power_aligned[i] > bull_power_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bear power negative and decreasing
                elif (bear_power_aligned[i] < 0 and 
                      i > start and bear_power_aligned[i] < bear_power_aligned[i-1]):
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

# Hypothesis: 6-hour Elder Ray Power (Bull/Bear) with 1-week volume confirmation.
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to detect trend strength.
# Volume confirmation ensures institutional participation in the move.
# Designed for 6h timeframe to target 50-150 trades over 4 years with low frequency.
# Works in both bull and bear markets by adapting to trend direction via EMA.

name = "6h_elderray1w_vol_v1"
timeframe = "6h"
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
    
    # 1-week EMA(13) for Elder Ray calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA calculation
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 13:
        ema_1w[12] = np.mean(close_1w[:13])
        for i in range(13, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (13 + 1)) + (ema_1w[i-1] * (11 / (13 + 1)))
    
    # Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = np.full(len(close_1w), np.nan)
    bear_power = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if not np.isnan(ema_1w[i]):
            bull_power[i] = df_1w['high'].iloc[i] - ema_1w[i]
            bear_power[i] = df_1w['low'].iloc[i] - ema_1w[i]
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    if len(vol_1w) >= 5:
        for i in range(4, len(vol_1w)):
            vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(13, 4)  # EMA needs 13, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear power turns positive (trend weakening) or stoploss
            if (bear_power_aligned[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull power turns negative (trend weakening) or stoploss
            if (bull_power_aligned[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: Bull power positive and increasing
                if (bull_power_aligned[i] > 0 and 
                    i > start and bull_power_aligned[i] > bull_power_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bear power negative and decreasing
                elif (bear_power_aligned[i] < 0 and 
                      i > start and bear_power_aligned[i] < bear_power_aligned[i-1]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals