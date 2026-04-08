#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_reversion_v1"
timeframe = "6h"
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
    
    # Get 1-day data for CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp = (high_1d + low_1d + close_1d) / 3.0
    # 20-period SMA of typical price
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # CCI calculation
    cci = (tp - sma_tp) / (0.015 * mad)
    # Replace inf/NaN from zero mad
    cci = np.where(mad == 0, 0, cci)
    
    # Align CCI to 6h timeframe (wait for daily bar to close)
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # Volume confirmation on 6h: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.3
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if CCI is not available
        if np.isnan(cci_aligned[i]):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (overbought reversal)
            if cci_aligned[i] < 100 and cci_aligned[i-1] >= 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (oversold reversal)
            if cci_aligned[i] > -100 and cci_aligned[i-1] <= -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: CCI crosses above -100 from oversold with volume
            if cci_aligned[i] > -100 and cci_aligned[i-1] <= -100:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below +100 from overbought with volume
            elif cci_aligned[i] < 100 and cci_aligned[i-1] >= 100:
                position = -1
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_reversion_v1"
timeframe = "6h"
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
    
    # Get 1-day data for CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp = (high_1d + low_1d + close_1d) / 3.0
    # 20-period SMA of typical price
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # CCI calculation
    cci = (tp - sma_tp) / (0.015 * mad)
    # Replace inf/NaN from zero mad
    cci = np.where(mad == 0, 0, cci)
    
    # Align CCI to 6h timeframe (wait for daily bar to close)
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # Volume confirmation on 6h: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.3
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if CCI is not available
        if np.isnan(cci_aligned[i]):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (overbought reversal)
            if cci_aligned[i] < 100 and cci_aligned[i-1] >= 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (oversold reversal)
            if cci_aligned[i] > -100 and cci_aligned[i-1] <= -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: CCI crosses above -100 from oversold with volume
            if cci_aligned[i] > -100 and cci_aligned[i-1] <= -100:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below +100 from overbought with volume
            elif cci_aligned[i] < 100 and cci_aligned[i-1] >= 100:
                position = -1
                signals[i] = -0.25
    
    return signals