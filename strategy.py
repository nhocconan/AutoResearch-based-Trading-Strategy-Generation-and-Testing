#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-day EMA(50) trend filter and volume confirmation.
# Uses daily EMA for trend direction (works in bull/bear markets), Donchian breakout for momentum entry,
# and volume spike to confirm genuine breakouts. Designed for 6h timeframe to target 50-150 trades
# over 4 years (12-37/year) with low frequency to minimize fee drag. EMA(50) provides smooth trend
# filter that adapts to changing market regimes, while Donchian(20) captures breakouts from
# consolidation zones. Volume confirmation ensures breakouts have institutional participation.

name = "6h_donchian20_1d_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA(50) for trend direction - HTF
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) with proper initialization
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        # Initialize with SMA of first 50 values
        ema_1d[49] = np.mean(close_1d[:50])
        # Calculate EMA for remaining values
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 0.03846) + (ema_1d[i-1] * 0.96154)  # alpha = 2/(50+1)
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian Channel (20-period) - LTF
    # Highest high and lowest low over last 20 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Daily volume average - HTF
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(9, len(vol_1d)):  # 10-day average
        vol_ma_1d[i] = np.mean(vol_1d[i-10:i])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 19)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss
            if (close[i] < ema_1d_aligned[i] or 
                close[i] < entry_price - 2.0 * (highest_high[i] - lowest_low[i])):  # ATR-based stop approximation
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss
            if (close[i] > ema_1d_aligned[i] or 
                close[i] > entry_price + 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries only with volume confirmation
            if volume_filter:
                # Long: price breaks above Donchian upper band AND above daily EMA
                if close[i] > highest_high[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian lower band AND below daily EMA
                elif close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i]:
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

# Hypothesis: 6-hour Donchian(20) breakout with 1-day EMA(50) trend filter and volume confirmation.
# Uses daily EMA for trend direction (works in bull/bear markets), Donchian breakout for momentum entry,
# and volume spike to confirm genuine breakouts. Designed for 6h timeframe to target 50-150 trades
# over 4 years (12-37/year) with low frequency to minimize fee drag. EMA(50) provides smooth trend
# filter that adapts to changing market regimes, while Donchian(20) captures breakouts from
# consolidation zones. Volume confirmation ensures breakouts have institutional participation.

name = "6h_donchian20_1d_ema50_vol_v1"
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
    
    # Daily EMA(50) for trend direction - HTF
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) with proper initialization
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        # Initialize with SMA of first 50 values
        ema_1d[49] = np.mean(close_1d[:50])
        # Calculate EMA for remaining values
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 0.03846) + (ema_1d[i-1] * 0.96154)  # alpha = 2/(50+1)
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian Channel (20-period) - LTF
    # Highest high and lowest low over last 20 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Daily volume average - HTF
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(9, len(vol_1d)):  # 10-day average
        vol_ma_1d[i] = np.mean(vol_1d[i-10:i])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 19)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss
            if (close[i] < ema_1d_aligned[i] or 
                close[i] < entry_price - 2.0 * (highest_high[i] - lowest_low[i])):  # ATR-based stop approximation
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss
            if (close[i] > ema_1d_aligned[i] or 
                close[i] > entry_price + 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries only with volume confirmation
            if volume_filter:
                # Long: price breaks above Donchian upper band AND above daily EMA
                if close[i] > highest_high[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian lower band AND below daily EMA
                elif close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals