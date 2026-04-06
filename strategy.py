#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_volume_volatility"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower bands
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):  # 20-period window
        upper_1d[i] = np.max(high_1d[i-19:i+1])
        lower_1d[i] = np.min(low_1d[i-19:i+1])
    
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 1-day ATR for volatility filter
    tr_1d = np.full(len(high_1d), np.nan)
    atr_1d = np.full(len(high_1d), np.nan)
    
    if len(high_1d) > 1:
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(high_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i],
                          abs(high_1d[i] - close_1d[i-1]),
                          abs(low_1d[i] - close_1d[i-1]))
    
    if len(high_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[1:14])
        for i in range(14, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 13, 4)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i] * 1.5
        
        # Volatility filter: ATR > 0.5% of price (avoid low volatility periods)
        volatility_filter = atr_1d_aligned[i] > close[i] * 0.005
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below lower band or stoploss
            if (close[i] < lower_1d_aligned[i] or 
                close[i] < entry_price - 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above upper band or stoploss
            if (close[i] > upper_1d_aligned[i] or 
                close[i] > entry_price + 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and volatility confirmation
            if volume_filter and volatility_filter:
                # Long: breakout above upper band
                if close[i] > upper_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below lower band
                elif close[i] < lower_1d_aligned[i]:
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

name = "12h_donchian20_1d_volume_volatility"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower bands
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):  # 20-period window
        upper_1d[i] = np.max(high_1d[i-19:i+1])
        lower_1d[i] = np.min(low_1d[i-19:i+1])
    
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 1-day ATR for volatility filter
    tr_1d = np.full(len(high_1d), np.nan)
    atr_1d = np.full(len(high_1d), np.nan)
    
    if len(high_1d) > 1:
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(high_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i],
                          abs(high_1d[i] - close_1d[i-1]),
                          abs(low_1d[i] - close_1d[i-1]))
    
    if len(high_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[1:14])
        for i in range(14, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 13, 4)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i] * 1.5
        
        # Volatility filter: ATR > 0.5% of price (avoid low volatility periods)
        volatility_filter = atr_1d_aligned[i] > close[i] * 0.005
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below lower band or stoploss
            if (close[i] < lower_1d_aligned[i] or 
                close[i] < entry_price - 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above upper band or stoploss
            if (close[i] > upper_1d_aligned[i] or 
                close[i] > entry_price + 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and volatility confirmation
            if volume_filter and volatility_filter:
                # Long: breakout above upper band
                if close[i] > upper_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below lower band
                elif close[i] < lower_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals