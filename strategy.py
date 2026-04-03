#!/usr/bin/env python3
"""
Experiment #2207: 6h Williams %R + 1d EMA Trend + Volume Spike
HYPOTHESIS: Williams %R extremes on 6h timeframe capture oversold/overbought conditions,
filtered by 1d EMA trend direction and volume confirmation. Works in both bull (buying dips in uptrend)
and bear (selling rallies in downtrend) markets. Uses strict thresholds to limit trades to 50-150 total.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2207_6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Williams %R(14), Volume MA(20) ===
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    denominator = highest_high - lowest_low
    # Avoid division by zero
    mask = denominator != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / denominator[mask]) * -100
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: Williams %R mean reversion ---
        if in_position:
            if position_side > 0:  # Long
                # Exit when Williams %R returns above -20 (overbought threshold)
                if williams_r[i] > -20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit when Williams %R returns below -80 (oversold threshold)
                if williams_r[i] < -80:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        trend_bias = trend_1d_aligned[i]
        volume_spike = vol_ratio[i] > 1.5  # Require 1.5x average volume
        
        if volume_spike:
            # Long entry: Williams %R oversold (< -80) AND 1d trend up
            if trend_bias > 0 and williams_r[i] < -80:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Williams %R overbought (> -20) AND 1d trend down
            elif trend_bias < 0 and williams_r[i] > -20:
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
Experiment #2207: 6h Williams %R + 1d EMA Trend + Volume Spike
HYPOTHESIS: Williams %R extremes on 6h timeframe capture oversold/overbought conditions,
filtered by 1d EMA trend direction and volume confirmation. Works in both bull (buying dips in uptrend)
and bear (selling rallies in downtrend) markets. Uses strict thresholds to limit trades to 50-150 total.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2207_6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Williams %R(14), Volume MA(20) ===
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    denominator = highest_high - lowest_low
    # Avoid division by zero
    mask = denominator != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / denominator[mask]) * -100
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: Williams %R mean reversion ---
        if in_position:
            if position_side > 0:  # Long
                # Exit when Williams %R returns above -20 (overbought threshold)
                if williams_r[i] > -20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit when Williams %R returns below -80 (oversold threshold)
                if williams_r[i] < -80:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        trend_bias = trend_1d_aligned[i]
        volume_spike = vol_ratio[i] > 1.5  # Require 1.5x average volume
        
        if volume_spike:
            # Long entry: Williams %R oversold (< -80) AND 1d trend up
            if trend_bias > 0 and williams_r[i] < -80:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Williams %R overbought (> -20) AND 1d trend down
            elif trend_bias < 0 and williams_r[i] > -20:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals