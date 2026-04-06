#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and daily volume confirmation.
# Uses 4h Supertrend (ATR=10, mult=3) for trend direction - works in bull via uptrends and bear via downtrends.
# Daily volume filter (current volume > 1.5x 20-day average) ensures momentum has participation.
# 1h RSI(14) with thresholds (30/70) for entry timing on pullbacks within the trend.
# Target: 80-150 total trades over 4 years (20-38/year) with session filter (08-20 UTC) to reduce noise.

name = "1h_supertrend_4h_volume_rsi_v1"
timeframe = "1h"
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
    
    # 4h Supertrend (ATR=10, mult=3)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for 4h
    atr_4h = np.full(len(close_4h), np.nan)
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    
    for i in range(10, len(close_4h)):
        if i == 10:
            atr_4h[i] = np.mean(tr_4h[1:11])
        else:
            atr_4h[i] = (atr_4h[i-1] * 9 + tr_4h[i]) / 10
    
    # Calculate Supertrend
    upperband_4h = np.full(len(close_4h), np.nan)
    lowerband_4h = np.full(len(close_4h), np.nan)
    for i in range(10, len(close_4h)):
        upperband_4h[i] = (high_4h[i] + low_4h[i]) / 2 + 3 * atr_4h[i]
        lowerband_4h[i] = (high_4h[i] + low_4h[i]) / 2 - 3 * atr_4h[i]
    
    supertrend_4h = np.full(len(close_4h), np.nan)
    direction_4h = np.full(len(close_4h), np.nan)  # 1 for uptrend, -1 for downtrend
    for i in range(10, len(close_4h)):
        if i == 10:
            supertrend_4h[i] = lowerband_4h[i]
            direction_4h[i] = 1
        else:
            if supertrend_4h[i-1] == upperband_4h[i-1]:
                if close_4h[i] <= upperband_4h[i]:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
                else:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
            else:
                if close_4h[i] >= lowerband_4h[i]:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
                else:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
    
    # Align Supertrend direction to 1h
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Daily volume filter (volume > 1.5x 20-day average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # 1h RSI(14)
    rsi = np.full(n, np.nan)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(direction_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(rsi[i]) or hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (rsi[i] > 70 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (rsi[i] < 30 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and 4h trend
            if volume_filter:
                # Long: RSI < 30 (pullback) in 4h uptrend
                if (rsi[i] < 30 and direction_4h_aligned[i] == 1):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 (pullback) in 4h downtrend
                elif (rsi[i] > 70 and direction_4h_aligned[i] == -1):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and daily volume confirmation.
# Uses 4h Supertrend (ATR=10, mult=3) for trend direction - works in bull via uptrends and bear via downtrends.
# Daily volume filter (current volume > 1.5x 20-day average) ensures momentum has participation.
# 1h RSI(14) with thresholds (30/70) for entry timing on pullbacks within the trend.
# Target: 80-150 total trades over 4 years (20-38/year) with session filter (08-20 UTC) to reduce noise.

name = "1h_supertrend_4h_volume_rsi_v1"
timeframe = "1h"
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
    
    # 4h Supertrend (ATR=10, mult=3)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for 4h
    atr_4h = np.full(len(close_4h), np.nan)
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    
    for i in range(10, len(close_4h)):
        if i == 10:
            atr_4h[i] = np.mean(tr_4h[1:11])
        else:
            atr_4h[i] = (atr_4h[i-1] * 9 + tr_4h[i]) / 10
    
    # Calculate Supertrend
    upperband_4h = np.full(len(close_4h), np.nan)
    lowerband_4h = np.full(len(close_4h), np.nan)
    for i in range(10, len(close_4h)):
        upperband_4h[i] = (high_4h[i] + low_4h[i]) / 2 + 3 * atr_4h[i]
        lowerband_4h[i] = (high_4h[i] + low_4h[i]) / 2 - 3 * atr_4h[i]
    
    supertrend_4h = np.full(len(close_4h), np.nan)
    direction_4h = np.full(len(close_4h), np.nan)  # 1 for uptrend, -1 for downtrend
    for i in range(10, len(close_4h)):
        if i == 10:
            supertrend_4h[i] = lowerband_4h[i]
            direction_4h[i] = 1
        else:
            if supertrend_4h[i-1] == upperband_4h[i-1]:
                if close_4h[i] <= upperband_4h[i]:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
                else:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
            else:
                if close_4h[i] >= lowerband_4h[i]:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
                else:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
    
    # Align Supertrend direction to 1h
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Daily volume filter (volume > 1.5x 20-day average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # 1h RSI(14)
    rsi = np.full(n, np.nan)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(direction_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(rsi[i]) or hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (rsi[i] > 70 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (rsi[i] < 30 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and 4h trend
            if volume_filter:
                # Long: RSI < 30 (pullback) in 4h uptrend
                if (rsi[i] < 30 and direction_4h_aligned[i] == 1):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 (pullback) in 4h downtrend
                elif (rsi[i] > 70 and direction_4h_aligned[i] == -1):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and daily volume confirmation.
# Uses 4h Supertrend (ATR=10, mult=3) for trend direction - works in bull via uptrends and bear via downtrends.
# Daily volume filter (current volume > 1.5x 20-day average) ensures momentum has participation.
# 1h RSI(14) with thresholds (30/70) for entry timing on pullbacks within the trend.
# Target: 80-150 total trades over 4 years (20-38/year) with session filter (08-20 UTC) to reduce noise.

name = "1h_supertrend_4h_volume_rsi_v1"
timeframe = "1h"
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
    
    # 4h Supertrend (ATR=10, mult=3)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for 4h
    atr_4h = np.full(len(close_4h), np.nan)
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    
    for i in range(10, len(close_4h)):
        if i == 10:
            atr_4h[i] = np.mean(tr_4h[1:11])
        else:
            atr_4h[i] = (atr_4h[i-1] * 9 + tr_4h[i]) / 10
    
    # Calculate Supertrend
    upperband_4h = np.full(len(close_4h), np.nan)
    lowerband_4h = np.full(len(close_4h), np.nan)
    for i in range(10, len(close_4h)):
        upperband_4h[i] = (high_4h[i] + low_4h[i]) / 2 + 3 * atr_4h[i]
        lowerband_4h[i] = (high_4h[i] + low_4h[i]) / 2 - 3 * atr_4h[i]
    
    supertrend_4h = np.full(len(close_4h), np.nan)
    direction_4h = np.full(len(close_4h), np.nan)  # 1 for uptrend, -1 for downtrend
    for i in range(10, len(close_4h)):
        if i == 10:
            supertrend_4h[i] = lowerband_4h[i]
            direction_4h[i] = 1
        else:
            if supertrend_4h[i-1] == upperband_4h[i-1]:
                if close_4h[i] <= upperband_4h[i]:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
                else:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
            else:
                if close_4h[i] >= lowerband_4h[i]:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
                else:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
    
    # Align Supertrend direction to 1h
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Daily volume filter (volume > 1.5x 20-day average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # 1h RSI(14)
    rsi = np.full(n, np.nan)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(direction_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(rsi[i]) or hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (rsi[i] > 70 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (rsi[i] < 30 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and 4h trend
            if volume_filter:
                # Long: RSI < 30 (pullback) in 4h uptrend
                if (rsi[i] < 30 and direction_4h_aligned[i] == 1):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 (pullback) in 4h downtrend
                elif (rsi[i] > 70 and direction_4h_aligned[i] == -1):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and daily volume confirmation.
# Uses 4h Supertrend (ATR=10, mult=3) for trend direction - works in bull via uptrends and bear via downtrends.
# Daily volume filter (current volume > 1.5x 20-day average) ensures momentum has participation.
# 1h RSI(14) with thresholds (30/70) for entry timing on pullbacks within the trend.
# Target: 80-150 total trades over 4 years (20-38/year) with session filter (08-20 UTC) to reduce noise.

name = "1h_supertrend_4h_volume_rsi_v1"
timeframe = "1h"
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
    
    # 4h Supertrend (ATR=10, mult=3)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for 4h
    atr_4h = np.full(len(close_4h), np.nan)
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    
    for i in range(10, len(close_4h)):
        if i == 10:
            atr_4h[i] = np.mean(tr_4h[1:11])
        else:
            atr_4h[i] = (atr_4h[i-1] * 9 + tr_4h[i]) / 10
    
    # Calculate Supertrend
    upperband_4h = np.full(len(close_4h), np.nan)
    lowerband_4h = np.full(len(close_4h), np.nan)
    for i in range(10, len(close_4h)):
        upperband_4h[i] = (high_4h[i] + low_4h[i]) / 2 + 3 * atr_4h[i]
        lowerband_4h[i] = (high_4h[i] + low_4h[i]) / 2 - 3 * atr_4h[i]
    
    supertrend_4h = np.full(len(close_4h), np.nan)
    direction_4h = np.full(len(close_4h), np.nan)  # 1 for uptrend, -1 for downtrend
    for i in range(10, len(close_4h)):
        if i == 10:
            supertrend_4h[i] = lowerband_4h[i]
            direction_4h[i] = 1
        else:
            if supertrend_4h[i-1] == upperband_4h[i-1]:
                if close_4h[i] <= upperband_4h[i]:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
                else:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
            else:
                if close_4h[i] >= lowerband_4h[i]:
                    supertrend_4h[i] = upperband_4h[i]
                    direction_4h[i] = -1
                else:
                    supertrend_4h[i] = lowerband_4h[i]
                    direction_4h[i] = 1
    
    # Align Supertrend direction to 1h
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Daily volume filter (volume > 1.5x 20-day average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # 1h RSI(14)
    rsi = np.full(n, np.nan)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(direction_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(rsi[i]) or hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (rsi[i] > 70 or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (rsi[i] < 30 or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and 4h trend
            if volume_filter:
                # Long: RSI < 30 (pullback) in 4h uptrend
                if (rsi[i] < 30 and direction_4h_aligned[i] == 1):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 (pullback) in 4h downtrend
                elif (rsi[i] > 70 and direction_4h_aligned[i] == -1):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and daily volume confirmation.
# Uses 4h Supertrend (ATR=10, mult=3) for trend direction - works in bull via uptrends and bear via downtrends.
# Daily volume filter (current volume > 1.5x 20-day average) ensures momentum has participation.
# 1h RSI(14) with thresholds (30/70) for entry timing on pullbacks within the trend.
# Target: 80-150 total trades over 4 years (20-38/year) with session filter (08-20 UTC) to reduce noise.

name = "1h_supertrend_4h_volume_rsi_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n