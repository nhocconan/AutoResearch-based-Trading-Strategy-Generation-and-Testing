#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily pivot levels from previous day (for 6h timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Camarilla: H = High, L = Low, C = Close of previous day
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    HLC_diff = (high_1d - low_1d)
    R4 = close_1d + HLC_diff * 1.1 / 2
    R3 = close_1d + HLC_diff * 1.1 / 4
    S3 = close_1d - HLC_diff * 1.1 / 4
    S4 = close_1d - HLC_diff * 1.1 / 2
    
    # Align pivot levels to 6h timeframe (shifted by 1 day for no look-ahead)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: 6h volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Range filter: only trade when price is between S3 and R3 (mean reversion zone)
    in_range = (close > S3_aligned) & (close < R3_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss (1.5x daily range)
        daily_range = (high_1d[i] - low_1d[i]) if not np.isnan(high_1d[i]) and not np.isnan(low_1d[i]) else 0
        if np.isnan(daily_range) or daily_range == 0:
            daily_range = high[i] - low[i]  # fallback to 6h range
        
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) or stoploss
            if (close[i] <= S3_aligned[i] or 
                close[i] < entry_price - 1.5 * daily_range):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) or stoploss
            if (close[i] >= R3_aligned[i] or 
                close[i] > entry_price + 1.5 * daily_range):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at extreme levels
            if volume_filter and in_range[i]:
                # Long: price touches or goes below S4 with rejection (close > S4)
                if close[i] <= S4_aligned[i] and close[i] > S4_aligned[i] * 0.999:  # touched S4
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price touches or goes above R4 with rejection (close < R4)
                elif close[i] >= R4_aligned[i] and close[i] < R4_aligned[i] * 1.001:  # touched R4
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

name = "6h_camarilla_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily pivot levels from previous day (for 6h timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Camarilla: H = High, L = Low, C = Close of previous day
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    HLC_diff = (high_1d - low_1d)
    R4 = close_1d + HLC_diff * 1.1 / 2
    R3 = close_1d + HLC_diff * 1.1 / 4
    S3 = close_1d - HLC_diff * 1.1 / 4
    S4 = close_1d - HLC_diff * 1.1 / 2
    
    # Align pivot levels to 6h timeframe (shifted by 1 day for no look-ahead)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: 6h volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Range filter: only trade when price is between S3 and R3 (mean reversion zone)
    in_range = (close > S3_aligned) & (close < R3_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss (1.5x daily range)
        daily_range = (high_1d[i] - low_1d[i]) if not np.isnan(high_1d[i]) and not np.isnan(low_1d[i]) else 0
        if np.isnan(daily_range) or daily_range == 0:
            daily_range = high[i] - low[i]  # fallback to 6h range
        
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) or stoploss
            if (close[i] <= S3_aligned[i] or 
                close[i] < entry_price - 1.5 * daily_range):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) or stoploss
            if (close[i] >= R3_aligned[i] or 
                close[i] > entry_price + 1.5 * daily_range):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at extreme levels
            if volume_filter and in_range[i]:
                # Long: price touches or goes below S4 with rejection (close > S4)
                if close[i] <= S4_aligned[i] and close[i] > S4_aligned[i] * 0.999:  # touched S4
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price touches or goes above R4 with rejection (close < R4)
                elif close[i] >= R4_aligned[i] and close[i] < R4_aligned[i] * 1.001:  # touched R4
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals