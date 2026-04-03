#!/usr/bin/env python3
"""
Experiment #1977: 4h Donchian Breakout + 1d Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant price moves. 
Filtering by 1d EMA trend ensures alignment with higher timeframe momentum. 
Volume spike (>2x 20-period average) confirms institutional participation. 
ATR-based stoploss (2.5x ATR) manages risk. 
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1977_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian channels and ATR ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume spike detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - balances return and drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 = long, -1 = short
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators (max of 20, 14, 20, 50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss: exit if price moves 2.5*ATR against position
            exit_signal = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if not exit_signal:
                if position_side > 0 and price <= low_roll_min[i]:
                    exit_signal = True
                elif position_side < 0 and price >= high_roll_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d uptrend
            if trend_1d_aligned[i] > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d downtrend
            elif trend_1d_aligned[i] < 0 and price < low_roll_min[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1977: 4h Donchian Breakout + 1d Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant price moves. 
Filtering by 1d EMA trend ensures alignment with higher timeframe momentum. 
Volume spike (>2x 20-period average) confirms institutional participation. 
ATR-based stoploss (2.5x ATR) manages risk. 
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1977_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian channels and ATR ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume spike detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - balances return and drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 = long, -1 = short
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators (max of 20, 14, 20, 50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss: exit if price moves 2.5*ATR against position
            exit_signal = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if not exit_signal:
                if position_side > 0 and price <= low_roll_min[i]:
                    exit_signal = True
                elif position_side < 0 and price >= high_roll_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d uptrend
            if trend_1d_aligned[i] > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d downtrend
            elif trend_1d_aligned[i] < 0 and price < low_roll_min[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1977: 4h Donchian Breakout + 1d Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant price moves. 
Filtering by 1d EMA trend ensures alignment with higher timeframe momentum. 
Volume spike (>2x 20-period average) confirms institutional participation. 
ATR-based stoploss (2.5x ATR) manages risk. 
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1977_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian channels and ATR ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume spike detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - balances return and drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 = long, -1 = short
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators (max of 20, 14, 20, 50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss: exit if price moves 2.5*ATR against position
            exit_signal = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if not exit_signal:
                if position_side > 0 and price <= low_roll_min[i]:
                    exit_signal = True
                elif position_side < 0 and price >= high_roll_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d uptrend
            if trend_1d_aligned[i] > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d downtrend
            elif trend_1d_aligned[i] < 0 and price < low_roll_min[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1977: 4h Donchian Breakout + 1d Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant price moves. 
Filtering by 1d EMA trend ensures alignment with higher timeframe momentum. 
Volume spike (>2x 20-period average) confirms institutional participation. 
ATR-based stoploss (2.5x ATR) manages risk. 
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1977_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian channels and ATR ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume spike detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - balances return and drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 = long, -1 = short
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators (max of 20, 14, 20, 50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss: exit if price moves 2.5*ATR against position
            exit_signal = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if not exit_signal:
                if position_side > 0 and price <= low_roll_min[i]:
                    exit_signal = True
                elif position_side < 0 and price >= high_roll_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d uptrend
            if trend_1d_aligned[i] > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d downtrend
            elif trend_1d_aligned[i] < 0 and price < low_roll_min[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #1977: 4h Donchian Breakout + 1d Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant price moves. 
Filtering by 1d EMA trend ensures alignment with higher timeframe momentum. 
Volume spike (>2x 20-period average) confirms institutional participation. 
ATR-based stoploss (2.5x ATR) manages risk. 
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1977_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian channels and ATR ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume spike detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - balances return and drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 = long, -1 = short
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators (max of 20, 14, 20, 50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss: exit if price moves 2.5*ATR against position
            exit_signal = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if not exit_signal:
                if position_side > 0 and price <= low_roll_min[i]:
                    exit_signal = True
                elif position_side < 0 and price >= high_roll_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d uptrend
            if trend_1d_aligned[i] > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d downtrend
            elif trend_1d_aligned[i] < 0 and price < low_roll_min[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #1977: 4h Donchian Breakout + 1d Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant price moves. 
Filtering by 1d EMA trend ensures alignment with higher timeframe momentum. 
Volume spike (>2x 20-period average) confirms institutional participation. 
ATR-based stoploss (2.5x ATR) manages risk. 
Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1977_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian channels and ATR ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] =