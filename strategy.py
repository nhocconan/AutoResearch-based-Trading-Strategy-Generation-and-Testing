#!/usr/bin/env python3
"""
Experiment #1667: 6h Elder Ray + 1d/1w Trend + Volume Spike
HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h identifies strong momentum in direction of higher timeframe trend (1d/1w). Volume spike (>2x MA) confirms institutional participation. This combination works in both bull and bear markets by only taking trades aligned with the dominant trend, reducing whipsaw. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing (0.25) to minimize fee drag and manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1667_6h_elder_ray_1d_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d and 1w data for trend filters (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA(50) for trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(ema_1d > 0, 1, -1)  # Simplified: price > EMA = uptrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1w EMA(20) for major trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_1w = np.where(ema_1w > 0, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13  # Negative values indicate selling pressure
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 13, 50)  # sufficient for volume MA, EMA13, EMA1d
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require alignment of both 1d and 1w trends
        long_trend = (trend_1d_aligned[i] > 0) and (trend_1w_aligned[i] > 0)
        short_trend = (trend_1d_aligned[i] < 0) and (trend_1w_aligned[i] < 0)
        
        # Volume confirmation: require significant spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        # Elder Ray conditions: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.percentile(bull_power[max(0,i-100):i+1], 70)
        strong_bear = bear_power[i] < 0 and bear_power[i] < np.percentile(bear_power[max(0,i-100):i+1], 30)
        
        if volume_spike:
            if long_trend and strong_bull:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_trend and strong_bear:
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
Experiment #1667: 6h Elder Ray + 1d/1w Trend + Volume Spike
HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h identifies strong momentum in direction of higher timeframe trend (1d/1w). Volume spike (>2x MA) confirms institutional participation. This combination works in both bull and bear markets by only taking trades aligned with the dominant trend, reducing whipsaw. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing (0.25) to minimize fee drag and manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1667_6h_elder_ray_1d_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d and 1w data for trend filters (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA(50) for trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(ema_1d > 0, 1, -1)  # Simplified: price > EMA = uptrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1w EMA(20) for major trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_1w = np.where(ema_1w > 0, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13  # Negative values indicate selling pressure
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 13, 50)  # sufficient for volume MA, EMA13, EMA1d
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require alignment of both 1d and 1w trends
        long_trend = (trend_1d_aligned[i] > 0) and (trend_1w_aligned[i] > 0)
        short_trend = (trend_1d_aligned[i] < 0) and (trend_1w_aligned[i] < 0)
        
        # Volume confirmation: require significant spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        # Elder Ray conditions: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.percentile(bull_power[max(0,i-100):i+1], 70)
        strong_bear = bear_power[i] < 0 and bear_power[i] < np.percentile(bear_power[max(0,i-100):i+1], 30)
        
        if volume_spike:
            if long_trend and strong_bull:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_trend and strong_bear:
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