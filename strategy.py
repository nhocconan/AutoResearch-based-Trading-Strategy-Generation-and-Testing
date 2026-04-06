#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour weekly Donchian(20) breakout with daily volume confirmation.
# Uses weekly Donchian channels to capture long-term trends, with daily volume spikes to filter.
# Designed for 12h timeframe to target 50-150 trades over 4 years with minimal false signals.
# Works in bull/bear markets via trend-following breakouts and volume confirmation.

name = "12h_weekly_donchian20_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for weekly calculation
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels
    highest_high_1w = np.full(len(high_1w), np.nan)
    lowest_low_1w = np.full(len(low_1w), np.nan)
    
    for i in range(19, len(high_1w)):
        highest_high_1w[i] = np.max(high_1w[i-19:i+1])
        lowest_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian to 12h timeframe (shifted by 1 week for no look-ahead)
    highest_high_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    lowest_low_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    
    # Daily volume confirmation: volume > 2x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day volume average
    vol_avg_20d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_20d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily volume average to 12h timeframe (shifted by 1 day)
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after sufficient warmup
        # Skip if required data not available
        if (np.isnan(highest_high_1w_aligned[i]) or 
            np.isnan(lowest_low_1w_aligned[i]) or 
            np.isnan(vol_avg_20d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 2x 20-day average
        # Get current daily volume (need to align current day's volume)
        current_day_volume = volume_1d[np.searchsorted(df_1d.index, prices.index[i])] if i < len(prices) else volume_1d[-1]
        # Simplified: use current 12h volume scaled to daily approximation
        volume_filter = volume[i] > (vol_avg_20d_aligned[i] * 2) if not np.isnan(vol_avg_20d_aligned[i]) else False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below weekly support or stoploss (2x ATR approximation)
            donch_width = highest_high_1w_aligned[i] - lowest_low_1w_aligned[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high_1w_aligned[i] - lowest_low_1w_aligned[i] + 0.001)
            
            if (close[i] < lowest_low_1w_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above weekly resistance or stoploss
            donch_width = highest_high_1w_aligned[i] - lowest_low_1w_aligned[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high_1w_aligned[i] - lowest_low_1w_aligned[i] + 0.001)
            
            if (close[i] > highest_high_1w_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: breakout above weekly resistance
                if (highest_high_1w_aligned[i] > highest_high_1w_aligned[i-1] and 
                    close[i] > highest_high_1w_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below weekly support
                elif (lowest_low_1w_aligned[i] < lowest_low_1w_aligned[i-1] and 
                      close[i] < lowest_low_1w_aligned[i-1]):
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

# Hypothesis: 12-hour weekly Donchian(20) breakout with daily volume confirmation.
# Uses weekly Donchian channels to capture long-term trends, with daily volume spikes to filter.
# Designed for 12h timeframe to target 50-150 trades over 4 years with minimal false signals.
# Works in bull/bear markets via trend-following breakouts and volume confirmation.

name = "12h_weekly_donchian20_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for weekly calculation
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels
    highest_high_1w = np.full(len(high_1w), np.nan)
    lowest_low_1w = np.full(len(low_1w), np.nan)
    
    for i in range(19, len(high_1w)):
        highest_high_1w[i] = np.max(high_1w[i-19:i+1])
        lowest_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian to 12h timeframe (shifted by 1 week for no look-ahead)
    highest_high_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    lowest_low_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    
    # Daily volume confirmation: volume > 2x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day volume average
    vol_avg_20d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_20d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily volume average to 12h timeframe (shifted by 1 day)
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after sufficient warmup
        # Skip if required data not available
        if (np.isnan(highest_high_1w_aligned[i]) or 
            np.isnan(lowest_low_1w_aligned[i]) or 
            np.isnan(vol_avg_20d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 2x 20-day average
        # Get current daily volume (need to align current day's volume)
        current_day_volume = volume_1d[np.searchsorted(df_1d.index, prices.index[i])] if i < len(prices) else volume_1d[-1]
        # Simplified: use current 12h volume scaled to daily approximation
        volume_filter = volume[i] > (vol_avg_20d_aligned[i] * 2) if not np.isnan(vol_avg_20d_aligned[i]) else False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below weekly support or stoploss (2x ATR approximation)
            donch_width = highest_high_1w_aligned[i] - lowest_low_1w_aligned[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high_1w_aligned[i] - lowest_low_1w_aligned[i] + 0.001)
            
            if (close[i] < lowest_low_1w_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above weekly resistance or stoploss
            donch_width = highest_high_1w_aligned[i] - lowest_low_1w_aligned[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high_1w_aligned[i] - lowest_low_1w_aligned[i] + 0.001)
            
            if (close[i] > highest_high_1w_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: breakout above weekly resistance
                if (highest_high_1w_aligned[i] > highest_high_1w_aligned[i-1] and 
                    close[i] > highest_high_1w_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below weekly support
                elif (lowest_low_1w_aligned[i] < lowest_low_1w_aligned[i-1] and 
                      close[i] < lowest_low_1w_aligned[i-1]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals