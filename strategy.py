#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14052_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA on 1w (20-period)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with volume confirmation and EMA trend filter
        # Long: price breaks above upper Donchian with volume > MA and price > 1w EMA
        # Short: price breaks below lower Donchian with volume > MA and price < 1w EMA
        
        breakout_up = close[i] > high_20[i-1]  # break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # break below previous period's low
        volume_confirm = volume[i] > vol_ma[i]
        trend_filter_long = close[i] > ema_1w_aligned[i]
        trend_filter_short = close[i] < ema_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_confirm and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * (high[i-1] - low[i-1]))  # Simple ATR approximation
            elif breakout_down and volume_confirm and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * (high[i-1] - low[i-1]))
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse breakout
            if close[i] <= stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse breakout
            if close[i] >= stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14052_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA on 1w (20-period)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with volume confirmation and EMA trend filter
        # Long: price breaks above upper Donchian with volume > MA and price > 1w EMA
        # Short: price breaks below lower Donchian with volume > MA and price < 1w EMA
        
        breakout_up = close[i] > high_20[i-1]  # break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # break below previous period's low
        volume_confirm = volume[i] > vol_ma[i]
        trend_filter_long = close[i] > ema_1w_aligned[i]
        trend_filter_short = close[i] < ema_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_confirm and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                # Calculate ATR for stop loss
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price - (2.0 * atr_val)
            elif breakout_down and volume_confirm and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price + (2.0 * atr_val)
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse breakout
            if close[i] <= stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse breakout
            if close[i] >= stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14052_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA on 1w (20-period)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with volume confirmation and EMA trend filter
        # Long: price breaks above upper Donchian with volume > MA and price > 1w EMA
        # Short: price breaks below lower Donchian with volume > MA and price < 1w EMA
        
        breakout_up = close[i] > high_20[i-1]  # break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # break below previous period's low
        volume_confirm = volume[i] > vol_ma[i]
        trend_filter_long = close[i] > ema_1w_aligned[i]
        trend_filter_short = close[i] < ema_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_confirm and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price - (2.0 * atr_val)
            elif breakout_down and volume_confirm and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price + (2.0 * atr_val)
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse breakout
            if close[i] <= stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse breakout
            if close[i] >= stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14052_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA on 1w (20-period)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with volume confirmation and EMA trend filter
        # Long: price breaks above upper Donchian with volume > MA and price > 1w EMA
        # Short: price breaks below lower Donchian with volume > MA and price < 1w EMA
        
        breakout_up = close[i] > high_20[i-1]  # break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # break below previous period's low
        volume_confirm = volume[i] > vol_ma[i]
        trend_filter_long = close[i] > ema_1w_aligned[i]
        trend_filter_short = close[i] < ema_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_confirm and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price - (2.0 * atr_val)
            elif breakout_down and volume_confirm and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price + (2.0 * atr_val)
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse breakout
            if close[i] <= stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse breakout
            if close[i] >= stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14052_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA on 1w (20-period)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with volume confirmation and EMA trend filter
        # Long: price breaks above upper Donchian with volume > MA and price > 1w EMA
        # Short: price breaks below lower Donchian with volume > MA and price < 1w EMA
        
        breakout_up = close[i] > high_20[i-1]  # break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # break below previous period's low
        volume_confirm = volume[i] > vol_ma[i]
        trend_filter_long = close[i] > ema_1w_aligned[i]
        trend_filter_short = close[i] < ema_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_confirm and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price - (2.0 * atr_val)
            elif breakout_down and volume_confirm and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price + (2.0 * atr_val)
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse breakout
            if close[i] <= stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse breakout
            if close[i] >= stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14052_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA on 1w (20-period)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for EMA)
    start = max(20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with volume confirmation and EMA trend filter
        # Long: price breaks above upper Donchian with volume > MA and price > 1w EMA
        # Short: price breaks below lower Donchian with volume > MA and price < 1w EMA
        
        breakout_up = close[i] > high_20[i-1]  # break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # break below previous period's low
        volume_confirm = volume[i] > vol_ma[i]
        trend_filter_long = close[i] > ema_1w_aligned[i]
        trend_filter_short = close[i] < ema_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_confirm and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price - (2.0 * atr_val)
            elif breakout_down and volume_confirm and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                atr_val = calculate_atr(high[:i+1], low[:i+1], close[:i+1], 14)[-1]
                stop_price = entry_price + (2.0 * atr_val)
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reverse breakout
            if close[i] <= stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reverse breakout
            if close[i] >= stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14052_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1