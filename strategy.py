#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + choppiness regime
# Long when: Price touches L3 support + volume > 2x average + CHOP > 61.8 (ranging market)
# Short when: Price touches H3 resistance + volume > 2x average + CHOP > 61.8
# Exit when: Price moves to opposite H3/L3 level or CHOP < 38.2 (trending market)
# Uses 1d Camarilla for structure, volume for confirmation, CHOP for regime filter
# Target: 80-160 trades over 4 years (20-40/year) by combining rare confluence

name = "4h_camarilla_1d_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: range = high - low
    # L3 = close - (range * 1.1/4)
    # H3 = close + (range * 1.1/4)
    # L4 = close - (range * 1.1/2)
    # H4 = close + (range * 1.1/2)
    range_1d = high_1d - low_1d
    L3 = close_1d - (range_1d * 1.1 / 4)
    H3 = close_1d + (range_1d * 1.1 / 4)
    L4 = close_1d - (range_1d * 1.1 / 2)
    H4 = close_1d + (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # Choppiness Index (CHOP) on 4h - using high, low, close
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index"""
        atr = []
        for i in range(len(high_arr)):
            if i == 0:
                atr.append(high_arr[i] - low_arr[i])
            else:
                tr = max(
                    high_arr[i] - low_arr[i],
                    abs(high_arr[i] - close_arr[i-1]),
                    abs(low_arr[i] - close_arr[i-1])
                )
                atr.append(tr)
        
        # Smooth ATR
        atr_smooth = pd.Series(atr).rolling(window=window, min_periods=window).mean().values
        
        # Calculate highest high and lowest low over window
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        # CHOP = 100 * log10(sum(atr)/ (highest_high - lowest_low)) / log10(window)
        range_hl = highest_high - lowest_low
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            chop = 100 * np.log10(sum_atr / range_hl) / np.log10(window)
            chop = np.where((range_hl == 0) | (sum_atr == 0), 50, chop)
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Price reaches H3 OR CHOP < 38.2 (trending market)
            if close[i] >= H3_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Price reaches L3 OR CHOP < 38.2 (trending market)
            if close[i] <= L3_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Price touches L3/H3 + volume spike + ranging market (CHOP > 61.8)
            if volume[i] > volume_threshold[i] and chop[i] > 61.8:
                # Long when price touches L3 support
                if close[i] <= L3_aligned[i] * 1.001:  # Small buffer for touching
                    signals[i] = 0.25
                    position = 1
                # Short when price touches H3 resistance
                elif close[i] >= H3_aligned[i] * 0.999:  # Small buffer for touching
                    signals[i] = -0.25
                    position = -1
    
    return signals
</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + choppiness regime
# Long when: Price touches L3 support + volume > 2x average + CHOP > 61.8 (ranging market)
# Short when: Price touches H3 resistance + volume > 2x average + CHOP > 61.8
# Exit when: Price moves to opposite H3/L3 level or CHOP < 38.2 (trending market)
# Uses 1d Camarilla for structure, volume for confirmation, CHOP for regime filter
# Target: 80-160 trades over 4 years (20-40/year) by combining rare confluence

name = "4h_camarilla_1d_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: range = high - low
    # L3 = close - (range * 1.1/4)
    # H3 = close + (range * 1.1/4)
    # L4 = close - (range * 1.1/2)
    # H4 = close + (range * 1.1/2)
    range_1d = high_1d - low_1d
    L3 = close_1d - (range_1d * 1.1 / 4)
    H3 = close_1d + (range_1d * 1.1 / 4)
    L4 = close_1d - (range_1d * 1.1 / 2)
    H4 = close_1d + (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # Choppiness Index (CHOP) on 4h - using high, low, close
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index"""
        atr = []
        for i in range(len(high_arr)):
            if i == 0:
                atr.append(high_arr[i] - low_arr[i])
            else:
                tr = max(
                    high_arr[i] - low_arr[i],
                    abs(high_arr[i] - close_arr[i-1]),
                    abs(low_arr[i] - close_arr[i-1])
                )
                atr.append(tr)
        
        # Smooth ATR
        atr_smooth = pd.Series(atr).rolling(window=window, min_periods=window).mean().values
        
        # Calculate highest high and lowest low over window
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        # CHOP = 100 * log10(sum(atr)/ (highest_high - lowest_low)) / log10(window)
        range_hl = highest_high - lowest_low
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            chop = 100 * np.log10(sum_atr / range_hl) / np.log10(window)
            chop = np.where((range_hl == 0) | (sum_atr == 0), 50, chop)
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Price reaches H3 OR CHOP < 38.2 (trending market)
            if close[i] >= H3_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Price reaches L3 OR CHOP < 38.2 (trending market)
            if close[i] <= L3_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Price touches L3/H3 + volume spike + ranging market (CHOP > 61.8)
            if volume[i] > volume_threshold[i] and chop[i] > 61.8:
                # Long when price touches L3 support
                if close[i] <= L3_aligned[i] * 1.001:  # Small buffer for touching
                    signals[i] = 0.25
                    position = 1
                # Short when price touches H3 resistance
                elif close[i] >= H3_aligned[i] * 0.999:  # Small buffer for touching
                    signals[i] = -0.25
                    position = -1
    
    return signals