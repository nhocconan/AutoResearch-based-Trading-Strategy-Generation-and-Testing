#!/usr/bin/env python3
"""
Experiment #3007: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h.
ADX(14) on 1d filters for trending markets (ADX>25) to avoid whipsaws in ranging markets.
Volume spike (>2.0x 20-period average) confirms momentum. Only take longs when
%R crosses above -50 from below in uptrend, shorts when %R crosses below -50 from above in downtrend.
This combination captures momentum in trending markets while avoiding false signals in chop.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3007_6h_williamsr_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or %R extreme reversal ---
        if in_position:
            # Exit if %R reverses sharply (> -20 for longs, < -80 for shorts)
            if position_side > 0 and williams_r[i] > -20:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            elif position_side < 0 and williams_r[i] < -80:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            # Exit if ADX drops below 20 (trend weakening)
            elif adx_1d_aligned[i] < 20:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) and strong trend (ADX>25)
        volume_spike = vol_ratio[i] > 2.0
        strong_trend = adx_1d_aligned[i] > 25
        
        if volume_spike and strong_trend:
            # Williams %R crossover signals
            wr_prev = williams_r[i-1]
            wr_curr = williams_r[i]
            
            # Long entry: %R crosses above -50 from below (momentum building)
            if wr_prev < -50 and wr_curr >= -50:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: %R crosses below -50 from above (momentum building)
            elif wr_prev > -50 and wr_curr <= -50:
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
Experiment #3007: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h.
ADX(14) on 1d filters for trending markets (ADX>25) to avoid whipsaws in ranging markets.
Volume spike (>2.0x 20-period average) confirms momentum. Only take longs when
%R crosses above -50 from below in uptrend, shorts when %R crosses below -50 from above in downtrend.
This combination captures momentum in trending markets while avoiding false signals in chop.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3007_6h_williamsr_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or %R extreme reversal ---
        if in_position:
            # Exit if %R reverses sharply (> -20 for longs, < -80 for shorts)
            if position_side > 0 and williams_r[i] > -20:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            elif position_side < 0 and williams_r[i] < -80:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            # Exit if ADX drops below 20 (trend weakening)
            elif adx_1d_aligned[i] < 20:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) and strong trend (ADX>25)
        volume_spike = vol_ratio[i] > 2.0
        strong_trend = adx_1d_aligned[i] > 25
        
        if volume_spike and strong_trend:
            # Williams %R crossover signals
            wr_prev = williams_r[i-1]
            wr_curr = williams_r[i]
            
            # Long entry: %R crosses above -50 from below (momentum building)
            if wr_prev < -50 and wr_curr >= -50:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: %R crosses below -50 from above (momentum building)
            elif wr_prev > -50 and wr_curr <= -50:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals