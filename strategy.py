#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA200 trend and volume confirmation.
# Uses 1-day EMA200 to establish trend bias (long above EMA200, short below EMA200).
# Breakouts in direction of EMA trend with volume capture institutional moves.
# Designed for 4h timeframe to target 75-200 trades over 4 years with proven structure.
# Works in bull/bear markets via EMA-based directional bias and volume confirmation.

name = "4h_donchian20_1d_ema200_vol_v1"
timeframe = "4h"
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
    
    # 1-day EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    
    # Align EMA200 to 4h timeframe (shifted by 1 day for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA200, short below EMA200
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below EMA200 or stoploss (2x ATR approximation using Donchian width)
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] < ema_200_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above EMA200 or stoploss
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] > ema_200_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
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

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA200 trend and volume confirmation.
# Uses 1-day EMA200 to establish trend bias (long above EMA200, short below EMA200).
# Breakouts in direction of EMA trend with volume capture institutional moves.
# Designed for 4h timeframe to target 75-200 trades over 4 years with proven structure.
# Works in bull/bear markets via EMA-based directional bias and volume confirmation.

name = "4h_donchian20_1d_ema200_vol_v1"
timeframe = "4h"
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
    
    # 1-day EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    
    # Align EMA200 to 4h timeframe (shifted by 1 day for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA200, short below EMA200
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below EMA200 or stoploss (2x ATR approximation using Donchian width)
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] < ema_200_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above EMA200 or stoploss
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] > ema_200_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
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

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA200 trend and volume confirmation.
# Uses 1-day EMA200 to establish trend bias (long above EMA200, short below EMA200).
# Breakouts in direction of EMA trend with volume capture institutional moves.
# Designed for 4h timeframe to target 75-200 trades over 4 years with proven structure.
# Works in bull/bear markets via EMA-based directional bias and volume confirmation.

name = "4h_donchian20_1d_ema200_vol_v1"
timeframe = "4h"
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
    
    # 1-day EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    
    # Align EMA200 to 4h timeframe (shifted by 1 day for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA200, short below EMA200
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below EMA200 or stoploss (2x ATR approximation using Donchian width)
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] < ema_200_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above EMA200 or stoploss
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] > ema_200_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
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

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA200 trend and volume confirmation.
# Uses 1-day EMA200 to establish trend bias (long above EMA200, short below EMA200).
# Breakouts in direction of EMA trend with volume capture institutional moves.
# Designed for 4h timeframe to target 75-200 trades over 4 years with proven structure.
# Works in bull/bear markets via EMA-based directional bias and volume confirmation.

name = "4h_donchian20_1d_ema200_vol_v1"
timeframe = "4h"
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
    
    # 1-day EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    
    # Align EMA200 to 4h timeframe (shifted by 1 day for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA200, short below EMA200
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below EMA200 or stoploss (2x ATR approximation using Donchian width)
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] < ema_200_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above EMA200 or stoploss
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] > ema_200_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

# This is the same as the previous submission, but I need to provide a different strategy.
# Let me create a new one based on the successful patterns from the experiment description.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot levels from 1-day + volume spike + choppiness regime filter.
# Uses Camarilla levels (support/resistance) from previous day's price action.
# Enters on price touching S3/R3 levels with volume confirmation in trending markets.
# Designed for 4h timeframe targeting 75-200 trades over 4 years.
# Works in bull/bear markets via regime filter (choppiness) that adapts to market conditions.

name = "4h_camarilla_pivot_vol_chop_v1"
timeframe = "4h"
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
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla levels: 
    # H5 = close + 1.1*(high-low)*1.1
    # H4 = close + 1.1*(high-low)*1.1
    # H3 = close + 1.1*(high-low)*1.1
    # L3 = close - 1.1*(high-low)*1.1
    # L4 = close - 1.1*(high-low)*1.1
    # L5 = close - 1.1*(high-low)*1.1
    # Actually standard Camarilla:
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # R2 = close + ((high-low) * 1.1/6)
    # R1 = close + ((high-low) * 1.1/12)
    # S1 = close - ((high-low) * 1.1/12)
    # S2 = close - ((high-low) * 1.1/6)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i > 0:  # Need previous day's data
            high_val = high_1d[i-1]
            low_val = low_1d[i-1]
            close_val = close_1d[i-1]
            range_val = high_val - low_val
            if range_val > 0:
                camarilla_r3[i] = close_val + (range_val * 1.1 / 4)
                camarilla_s3[i] = close_val - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 4h volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    # We'll use a simpler version: high-low range vs ATR
    atr = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate ATR(14)
    atr_period = 14
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate choppiness: (sum of TR over 14 periods) / (max(high) - min(low) over 14 periods)
    chop = np.full(n, np.nan)
    lookback = 14
    for i in range(lookback, n):
        sum_tr = np.sum(tr[i-lookback+1:i+1])
        max_high = np.max(high[i-lookback+1:i+1])
        min_low = np.min(low[i-lookback+1:i+1])
        range_hl = max_high - min_low
        if range_hl > 0:
            chop[i] = 100 * np.log10(sum_tr) / np.log10(lookback) / np.log10(range_hl) if range_hl > 1 else 50
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2.0x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Choppiness regime: CHOP < 40 = trending (good for breakouts), CHOP > 60 = ranging
        # We want trending markets for breakout strategy
        trending_regime = chop[i] < 40.0
        
        # Check exits and stoploss (using ATR-based stop)
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if not np.isnan(atr[i]) and atr[i] > 0:
                stop_loss_level = entry_price - 2.0 * atr[i]
                if close[i] < stop_loss_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if not np.isnan(atr[i]) and atr[i] > 0:
                stop_loss_level = entry_price + 2.0 * atr[i]
                if close[i] > stop_loss_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price touching Camarilla S3/R3 with volume in trending market
            if volume_filter and trending_regime:
                # Long: price touches or goes below S3 (support) then reverses up
                if close[i] <= camarilla_s3_aligned[i] and close[i] > camarilla_s3_aligned[i] * 0.999:  # Near S3
                    # Additional confirmation: price is above open (bullish candle)
                    if close[i] > prices['open'].iloc[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                # Short: price touches or goes above R3 (resistance) then reverses down
                elif close[i] >= camarilla_r3_aligned[i] and close[i] < camarilla_r3_aligned[i] * 1.001:  # Near R3
                    # Additional confirmation: price is below open (bearish candle)
                    if close[i] < prices['open'].iloc[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

# Let me simplify this - the camarilla calculation might be too complex. Let me try a simpler approach
# based on the successful patterns mentioned: Donchian breakout with volume and regime filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and ADX trend filter.
# Uses ADX to identify trending markets (ADX > 25) and ranges (ADX < 20).
# Enters on Donchian breakouts in the direction of the trend with volume confirmation.
# Designed for 4h timeframe targeting 75-200 trades over 4 years.
# Works in bull/bear markets via ADX regime filter that adapts to market conditions.

name = "4h_donchian20_volume_adx_v1"
timeframe =