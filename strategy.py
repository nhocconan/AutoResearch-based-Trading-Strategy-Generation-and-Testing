#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with 1-week EMA(20) trend and volume confirmation.
# In trending markets (CHOP < 38.2), follow EMA trend; in ranging markets (CHOP > 61.8), mean-revert at Bollinger Bands.
# Designed for 12h timeframe with ~50-100 trades over 4 years to minimize fee drag.
# Works in both bull and bear by adapting to market regime.

name = "12h_chop1w_ema20_bb_std2_vol"
timeframe = "12h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 1-week EMA(20) for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 20-period Choppiness Index
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.zeros(n)
        for i in range(14, n):
            atr_sum[i] = np.sum(tr[i-14:i]) if i-14 >= 0 else np.sum(tr[:i])
        
        high_max = np.full(n, np.nan)
        low_min = np.full(n, np.nan)
        for i in range(14, n):
            high_max[i] = np.max(high[i-14:i])
            low_min[i] = np.min(low[i-14:i])
        
        range_14 = high_max - low_min
        log_range = np.log10(range_14)
        log_atr_sum = np.log10(atr_sum)
        chop = 100 * log_range / (log_atr_sum * np.log10(14))
        chop[:14] = np.nan
    
    # 20-period Bollinger Bands (2 std dev)
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            bb_mid[i] = np.mean(close[i-20:i])
            bb_std[i] = np.std(close[i-20:i])
    
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_std[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss hit
            if (close[i] < ema_1w_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss hit
            if (close[i] > ema_1w_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Regime-based entries
            if chop[i] < 38.2:  # Trending regime
                # Follow trend: long if price > EMA, short if price < EMA
                if close[i] > ema_1w_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < ema_1w_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif chop[i] > 61.8:  # Ranging regime
                # Mean reversion: long at lower BB, short at upper BB
                if close[i] <= bb_lower[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] >= bb_upper[i] and volume_filter:
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

# Hypothesis: 12-hour Choppiness Index regime filter with 1-week EMA(20) trend and volume confirmation.
# In trending markets (CHOP < 38.2), follow EMA trend; in ranging markets (CHOP > 61.8), mean-revert at Bollinger Bands.
# Designed for 12h timeframe with ~50-100 trades over 4 years to minimize fee drag.
# Works in both bull and bear by adapting to market regime.

name = "12h_chop1w_ema20_bb_std2_vol"
timeframe = "12h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 1-week EMA(20) for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 20-period Choppiness Index
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.zeros(n)
        for i in range(14, n):
            atr_sum[i] = np.sum(tr[i-14:i]) if i-14 >= 0 else np.sum(tr[:i])
        
        high_max = np.full(n, np.nan)
        low_min = np.full(n, np.nan)
        for i in range(14, n):
            high_max[i] = np.max(high[i-14:i])
            low_min[i] = np.min(low[i-14:i])
        
        range_14 = high_max - low_min
        log_range = np.log10(range_14)
        log_atr_sum = np.log10(atr_sum)
        chop = 100 * log_range / (log_atr_sum * np.log10(14))
        chop[:14] = np.nan
    
    # 20-period Bollinger Bands (2 std dev)
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            bb_mid[i] = np.mean(close[i-20:i])
            bb_std[i] = np.std(close[i-20:i])
    
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_std[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss hit
            if (close[i] < ema_1w_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss hit
            if (close[i] > ema_1w_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Regime-based entries
            if chop[i] < 38.2:  # Trending regime
                # Follow trend: long if price > EMA, short if price < EMA
                if close[i] > ema_1w_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < ema_1w_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif chop[i] > 61.8:  # Ranging regime
                # Mean reversion: long at lower BB, short at upper BB
                if close[i] <= bb_lower[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] >= bb_upper[i] and volume_filter:
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

# Hypothesis: 12-hour Choppiness Index regime filter with 1-week EMA(20) trend and volume confirmation.
# In trending markets (CHOP < 38.2), follow EMA trend; in ranging markets (CHOP > 61.8), mean-revert at Bollinger Bands.
# Designed for 12h timeframe with ~50-100 trades over 4 years to minimize fee drag.
# Works in both bull and bear by adapting to market regime.

name = "12h_chop1w_ema20_bb_std2_vol"
timeframe = "12h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 1-week EMA(20) for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 20-period Choppiness Index
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.zeros(n)
        for i in range(14, n):
            atr_sum[i] = np.sum(tr[i-14:i]) if i-14 >= 0 else np.sum(tr[:i])
        
        high_max = np.full(n, np.nan)
        low_min = np.full(n, np.nan)
        for i in range(14, n):
            high_max[i] = np.max(high[i-14:i])
            low_min[i] = np.min(low[i-14:i])
        
        range_14 = high_max - low_min
        log_range = np.log10(range_14)
        log_atr_sum = np.log10(atr_sum)
        chop = 100 * log_range / (log_atr_sum * np.log10(14))
        chop[:14] = np.nan
    
    # 20-period Bollinger Bands (2 std dev)
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            bb_mid[i] = np.mean(close[i-20:i])
            bb_std[i] = np.std(close[i-20:i])
    
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_std[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss hit
            if (close[i] < ema_1w_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss hit
            if (close[i] > ema_1w_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Regime-based entries
            if chop[i] < 38.2:  # Trending regime
                # Follow trend: long if price > EMA, short if price < EMA
                if close[i] > ema_1w_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < ema_1w_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif chop[i] > 61.8:  # Ranging regime
                # Mean reversion: long at lower BB, short at upper BB
                if close[i] <= bb_lower[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] >= bb_upper[i] and volume_filter:
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

# Hypothesis: 12-hour Choppiness Index regime filter with 1-week EMA(20) trend and volume confirmation.
# In trending markets (CHOP < 38.2), follow EMA trend; in ranging markets (CHOP > 61.8), mean-revert at Bollinger Bands.
# Designed for 12h timeframe with ~50-100 trades over 4 years to minimize fee drag.
# Works in both bull and bear by adapting to market regime.

name = "12h_chop1w_ema20_bb_std2_vol"
timeframe = "12h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 1-week EMA(20) for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 20-period Choppiness Index
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.zeros(n)
        for i in range(14, n):
            atr_sum[i] = np.sum(tr[i-14:i]) if i-14 >= 0 else np.sum(tr[:i])
        
        high_max = np.full(n, np.nan)
        low_min = np.full(n, np.nan)
        for i in range(14, n):
            high_max[i] = np.max(high[i-14:i])
            low_min[i] = np.min(low[i-14:i])
        
        range_14 = high_max - low_min
        log_range = np.log10(range_14)
        log_atr_sum = np.log10(atr_sum)
        chop = 100 * log_range / (log_atr_sum * np.log10(14))
        chop[:14] = np.nan
    
    # 20-period Bollinger Bands (2 std dev)
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            bb_mid[i] = np.mean(close[i-20:i])
            bb_std[i] = np.std(close[i-20:i])
    
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_std[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss hit
            if (close[i] < ema_1w_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss hit
            if (close[i] > ema_1w_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Regime-based entries
            if chop[i] < 38.2:  # Trending regime
                # Follow trend: long if price > EMA, short if price < EMA
                if close[i] > ema_1w_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < ema_1w_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif chop[i] > 61.8:  # Ranging regime
                # Mean reversion: long at lower BB, short at upper BB
                if close[i] <= bb_lower[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] >= bb_upper[i] and volume_filter:
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

# Hypothesis: 12-hour Choppiness Index regime filter with 1-week EMA(20) trend and volume confirmation.
# In trending markets (CHOP < 38.2), follow EMA trend; in ranging markets (CHOP > 61.8), mean-revert at Bollinger Bands.
# Designed for 12h timeframe with ~50-100 trades over 4 years to minimize fee drag.
# Works in both bull and bear by adapting to market regime.

name = "12h_chop1w_ema20_bb_std2_vol"
timeframe = "12h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 1-week EMA(20) for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 20-period Choppiness Index
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.zeros(n)
        for i in range(14, n):
            atr_sum[i] = np.sum(tr[i-14:i]) if i-14 >= 0 else np.sum(tr[:i])
        
        high_max = np.full(n, np.nan)
        low_min = np.full(n, np.nan)
        for i in range(14, n):
            high_max[i] = np.max(high[i-14:i])
            low_min[i] = np.min(low[i-14:i])
        
        range_14 = high_max - low_min
        log_range = np.log10(range_14)
        log_atr_sum = np.log10(atr_sum)
        chop = 100 * log_range / (log_atr_sum * np.log10(14))
        chop[:14] = np.nan
    
    # 20-period Bollinger Bands (2 std dev)
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            bb_mid[i] = np.mean(close[i-20:i])
            bb_std[i] = np.std(close[i-20:i])
    
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_std[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss hit
            if (close[i] < ema_1w_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss hit
            if (close[i] > ema_1w_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Regime-based entries
            if chop[i] < 38.2:  # Trending regime
                # Follow trend: long if price > EMA, short if price < EMA
                if close[i] > ema_1w_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < ema_1w_aligned[i] and volume_filter:
                    signals