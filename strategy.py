#!/usr/bin/env python3
"""
6h Chandelier Exit with Parabolic SAR Trend Filter
Hypothesis: Chandelier Exit provides volatility-based trailing stops while Parabolic SAR
identifies trend direction. This combination avoids whipsaws in ranging markets and
captures trends in both bull and bear markets. Uses 1d Parabolic SAR for higher timeframe
trend context. Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14459_6h_chandelier_psar_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Parabolic SAR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize PSAR arrays
    psar = np.zeros_like(close_1d)
    bull = np.ones_like(close_1d)  # True for bullish trend
    af = np.zeros_like(close_1d)
    ep = np.zeros_like(close_1d)
    
    # Set initial values
    psar[0] = low_1d[0]
    af[0] = af_start
    ep[0] = high_1d[0]
    
    # Calculate Parabolic SAR
    for i in range(1, len(close_1d)):
        if bull[i-1]:  # was bullish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if low_1d[i] < psar[i]:  # trend reversal to bearish
                bull[i] = False
                psar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = low_1d[i]
            else:
                bull[i] = True
                if high_1d[i] > ep[i-1]:
                    ep[i] = high_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # was bearish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if high_1d[i] > psar[i]:  # trend reversal to bullish
                bull[i] = True
                psar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = high_1d[i]
            else:
                bull[i] = False
                if low_1d[i] < ep[i-1]:
                    ep[i] = low_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Align PSAR to 6h timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    bull_aligned = align_htf_to_ltf(prices, df_1d, bull.astype(float))
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Chandelier Exit parameters
    atr_period = 22
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Chandelier Exit
    # Long exit: highest high - multiplier * ATR
    # Short exit: lowest low + multiplier * ATR
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=1).min().values
    
    chandelier_long_exit = highest_high - multiplier * atr
    chandelier_short_exit = lowest_low + multiplier * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = atr_period
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(psar_aligned[i]) or np.isnan(bull_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(chandelier_long_exit[i]) or
            np.isnan(chandelier_short_exit[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price hits Chandelier Exit long OR trend turns bearish
            if (close[i] <= chandelier_long_exit[i] or bull_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price hits Chandelier Exit short OR trend turns bullish
            if (close[i] >= chandelier_short_exit[i] or bull_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: trend direction + price near Chandelier Exit
            # Long: bullish trend and price above long exit
            # Short: bearish trend and price below short exit
            long_setup = (bull_aligned[i] > 0.5 and close[i] > chandelier_long_exit[i])
            short_setup = (bull_aligned[i] < 0.5 and close[i] < chandelier_short_exit[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Chandelier Exit with Parabolic SAR Trend Filter
Hypothesis: Chandelier Exit provides volatility-based trailing stops while Parabolic SAR
identifies trend direction. This combination avoids whipsaws in ranging markets and
captures trends in both bull and bear markets. Uses 1d Parabolic SAR for higher timeframe
trend context. Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14459_6h_chandelier_psar_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Parabolic SAR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize PSAR arrays
    psar = np.zeros_like(close_1d)
    bull = np.ones_like(close_1d)  # True for bullish trend
    af = np.zeros_like(close_1d)
    ep = np.zeros_like(close_1d)
    
    # Set initial values
    psar[0] = low_1d[0]
    af[0] = af_start
    ep[0] = high_1d[0]
    
    # Calculate Parabolic SAR
    for i in range(1, len(close_1d)):
        if bull[i-1]:  # was bullish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if low_1d[i] < psar[i]:  # trend reversal to bearish
                bull[i] = False
                psar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = low_1d[i]
            else:
                bull[i] = True
                if high_1d[i] > ep[i-1]:
                    ep[i] = high_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # was bearish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            if high_1d[i] > psar[i]:  # trend reversal to bullish
                bull[i] = True
                psar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = high_1d[i]
            else:
                bull[i] = False
                if low_1d[i] < ep[i-1]:
                    ep[i] = low_1d[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Align PSAR to 6h timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    bull_aligned = align_htf_to_ltf(prices, df_1d, bull.astype(float))
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Chandelier Exit parameters
    atr_period = 22
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Chandelier Exit
    # Long exit: highest high - multiplier * ATR
    # Short exit: lowest low + multiplier * ATR
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=1).min().values
    
    chandelier_long_exit = highest_high - multiplier * atr
    chandelier_short_exit = lowest_low + multiplier * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = atr_period
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(psar_aligned[i]) or np.isnan(bull_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(chandelier_long_exit[i]) or
            np.isnan(chandelier_short_exit[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price hits Chandelier Exit long OR trend turns bearish
            if (close[i] <= chandelier_long_exit[i] or bull_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price hits Chandelier Exit short OR trend turns bullish
            if (close[i] >= chandelier_short_exit[i] or bull_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: trend direction + price near Chandelier Exit
            # Long: bullish trend and price above long exit
            # Short: bearish trend and price below short exit
            long_setup = (bull_aligned[i] > 0.5 and close[i] > chandelier_long_exit[i])
            short_setup = (bull_aligned[i] < 0.5 and close[i] < chandelier_short_exit[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>