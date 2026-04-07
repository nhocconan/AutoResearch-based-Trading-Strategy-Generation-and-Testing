#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_1d_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla pivot formula
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Resistance and support levels
    r4 = pivot + (range_prev * 1.1 / 2)
    r3 = pivot + (range_prev * 1.1 / 4)
    r2 = pivot + (range_prev * 1.1 / 6)
    r1 = pivot + (range_prev * 1.1 / 12)
    s1 = pivot - (range_prev * 1.1 / 12)
    s2 = pivot - (range_prev * 1.1 / 6)
    s3 = pivot - (range_prev * 1.1 / 4)
    s4 = pivot - (range_prev * 1.1 / 2)
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 12h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection (volume > 2x 20-period average)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below S1 or trend changes
            elif close[i] < s1_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above R1 or trend changes
            elif close[i] > r1_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Camarilla levels with volume confirmation
            # Long at S1 with volume spike
            if (close[i] <= s1_aligned[i] * 1.001 and  # allow small tolerance
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema50[i]):  # only long in uptrend
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short at R1 with volume spike
            elif (close[i] >= r1_aligned[i] * 0.999 and  # allow small tolerance
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema50[i]):  # only short in downtrend
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla pivot formula
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Resistance and support levels
    r1 = pivot + (range_prev * 1.1 / 12)
    s1 = pivot - (range_prev * 1.1 / 12)
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection (volume > 2x 20-period average)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below S1 or trend changes
            elif close[i] < s1_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above R1 or trend changes
            elif close[i] > r1_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Camarilla levels with volume confirmation
            # Long at S1 with volume spike
            if (close[i] <= s1_aligned[i] * 1.001 and  # allow small tolerance
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema50[i]):  # only long in uptrend
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short at R1 with volume spike
            elif (close[i] >= r1_aligned[i] * 0.999 and  # allow small tolerance
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema50[i]):  # only short in downtrend
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals