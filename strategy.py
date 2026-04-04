#!/usr/bin/env python3
"""
Experiment #2929: 4h Donchian20 + 1d EMA50 + Volume Spike (Tight)
HYPOTHESIS: Donchian(20) breakouts on 4h capture trends. 1d EMA50 provides trend filter: only take longs when close > EMA50, shorts when close < EMA50. Volume spike (>2.0x 20-period average) confirms strength. This combination reduces false breakouts in choppy markets. 4h timeframe targets 19-50 trades/year. Discrete size 0.25 minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2929_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 4h timeframe (shifted by 1 for completed bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Donchian mean reversion ---
        if signals[i-1] > 0:  # Long position
            if price <= highest_high[i]:  # Re-entered channel
                signals[i] = 0.0
            else:
                signals[i] = SIZE
        elif signals[i-1] < 0:  # Short position
            if price >= lowest_low[i]:  # Re-entered channel
                signals[i] = 0.0
            else:
                signals[i] = -SIZE
        else:  # No position
            # --- New Position Entry Logic ---
            # Require volume spike (> 2.0x average) for confirmation
            volume_spike = vol_ratio[i] > 2.0
            
            if volume_spike:
                # Get 1d EMA50 trend filter
                price_vs_ema = price - ema_1d_aligned[i]
                
                # Long entry: price breaks above Donchian high with bullish 1d trend
                if price > highest_high[i] and price_vs_ema > 0:
                    signals[i] = SIZE
                # Short entry: price breaks below Donchian low with bearish 1d trend
                elif price < lowest_low[i] and price_vs_ema < 0:
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #2929: 4h Donchian20 + 1d EMA50 + Volume Spike (Tight)
HYPOTHESIS: Donchian(20) breakouts on 4h capture trends. 1d EMA50 provides trend filter: only take longs when close > EMA50, shorts when close < EMA50. Volume spike (>2.0x 20-period average) confirms strength. This combination reduces false breakouts in choppy markets. 4h timeframe targets 19-50 trades/year. Discrete size 0.25 minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2929_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 4h timeframe (shifted by 1 for completed bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Donchian mean reversion ---
        if signals[i-1] > 0:  # Long position
            if price <= highest_high[i]:  # Re-entered channel
                signals[i] = 0.0
            else:
                signals[i] = SIZE
        elif signals[i-1] < 0:  # Short position
            if price >= lowest_low[i]:  # Re-entered channel
                signals[i] = 0.0
            else:
                signals[i] = -SIZE
        else:  # No position
            # --- New Position Entry Logic ---
            # Require volume spike (> 2.0x average) for confirmation
            volume_spike = vol_ratio[i] > 2.0
            
            if volume_spike:
                # Get 1d EMA50 trend filter
                price_vs_ema = price - ema_1d_aligned[i]
                
                # Long entry: price breaks above Donchian high with bullish 1d trend
                if price > highest_high[i] and price_vs_ema > 0:
                    signals[i] = SIZE
                # Short entry: price breaks below Donchian low with bearish 1d trend
                elif price < lowest_low[i] and price_vs_ema < 0:
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals