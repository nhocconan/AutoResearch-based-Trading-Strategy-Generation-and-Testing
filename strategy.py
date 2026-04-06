#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_cross_volume_trend_v1"
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
    
    # EMA(21) and EMA(55) for trend
    ema_21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False).mean().values
    
    # Daily trend filter - load once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation - volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):  # Wait for EMA(55) to stabilize
        # Skip if required data not available
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: EMA(21) crosses below EMA(55) OR price below daily EMA
            if ema_21[i] < ema_55[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: EMA(21) crosses above EMA(55) OR price above daily EMA
            if ema_21[i] > ema_55[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: EMA crossover + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if ema_21[i] > ema_55[i] and close[i] > ema_50_1d_aligned[i]:
                    # Bullish crossover with daily trend confirmation
                    signals[i] = 0.25
                    position = 1
                elif ema_21[i] < ema_55[i] and close[i] < ema_50_1d_aligned[i]:
                    # Bearish crossover with daily trend confirmation
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_cross_volume_trend_v1"
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
    
    # EMA(21) and EMA(55) for trend
    ema_21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False).mean().values
    
    # Daily trend filter - load once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation - volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):  # Wait for EMA(55) to stabilize
        # Skip if required data not available
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: EMA(21) crosses below EMA(55) OR price below daily EMA
            if ema_21[i] < ema_55[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: EMA(21) crosses above EMA(55) OR price above daily EMA
            if ema_21[i] > ema_55[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: EMA crossover + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if ema_21[i] > ema_55[i] and close[i] > ema_50_1d_aligned[i]:
                    # Bullish crossover with daily trend confirmation
                    signals[i] = 0.25
                    position = 1
                elif ema_21[i] < ema_55[i] and close[i] < ema_50_1d_aligned[i]:
                    # Bearish crossover with daily trend confirmation
                    signals[i] = -0.25
                    position = -1
    
    return signals