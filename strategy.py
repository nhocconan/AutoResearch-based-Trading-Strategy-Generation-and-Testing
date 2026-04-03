#!/usr/bin/env python3
"""
Experiment #1970: 1d Donchian(20) breakout + 1w trend filter + volume confirmation
HYPOTHESIS: Daily Donchian channel breakouts with weekly trend alignment and volume spike provide
high-probability entries that work in both bull and bear markets. Weekly EMA(21) filter ensures
we only trade with the dominant higher timeframe trend, reducing whipsaw. Volume confirmation
ensures breakouts have institutional participation. Target: 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1970_1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        dc_upper[i] = np.max(high[i-lookback:i])
        dc_lower[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20) + 21  # Donchian(20) + volume MA(20) + weekly EMA(21)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # === Exit Logic: ATR-based stoploss (2x ATR) ===
        if in_position:
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                
                # Simple ATR calculation (could be optimized but OK for 1d)
                atr_sum = 0.0
                for j in range(14):
                    idx = i - j
                    tr1_j = high[idx] - low[idx]
                    tr2_j = abs(high[idx] - close[idx-1]) if idx > 0 else 0
                    tr3_j = abs(low[idx] - close[idx-1]) if idx > 0 else 0
                    tr_j = max(tr1_j, tr2_j, tr3_j)
                    atr_sum += tr_j
                atr = atr_sum / 14.0
            else:
                atr = 0.0
            
            # Stoploss: exit if price moves 2*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > entry_price + 2.0 * atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # === New Position Entry Logic ===
        # Require weekly trend alignment
        weekly_trend = trend_1w_aligned[i]
        
        # Volume confirmation: require significant volume spike
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND weekly trend up
            if weekly_trend > 0 and price > dc_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND weekly trend down
            elif weekly_trend < 0 and price < dc_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1970: 1d Donchian(20) breakout + 1w trend filter + volume confirmation
HYPOTHESIS: Daily Donchian channel breakouts with weekly trend alignment and volume spike provide
high-probability entries that work in both bull and bear markets. Weekly EMA(21) filter ensures
we only trade with the dominant higher timeframe trend, reducing whipsaw. Volume confirmation
ensures breakouts have institutional participation. Target: 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1970_1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        dc_upper[i] = np.max(high[i-lookback:i])
        dc_lower[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20) + 21  # Donchian(20) + volume MA(20) + weekly EMA(21)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # === Exit Logic: ATR-based stoploss (2x ATR) ===
        if in_position:
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                
                # Simple ATR calculation (could be optimized but OK for 1d)
                atr_sum = 0.0
                for j in range(14):
                    idx = i - j
                    tr1_j = high[idx] - low[idx]
                    tr2_j = abs(high[idx] - close[idx-1]) if idx > 0 else 0
                    tr3_j = abs(low[idx] - close[idx-1]) if idx > 0 else 0
                    tr_j = max(tr1_j, tr2_j, tr3_j)
                    atr_sum += tr_j
                atr = atr_sum / 14.0
            else:
                atr = 0.0
            
            # Stoploss: exit if price moves 2*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > entry_price + 2.0 * atr:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # === New Position Entry Logic ===
        # Require weekly trend alignment
        weekly_trend = trend_1w_aligned[i]
        
        # Volume confirmation: require significant volume spike
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND weekly trend up
            if weekly_trend > 0 and price > dc_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND weekly trend down
            elif weekly_trend < 0 and price < dc_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>