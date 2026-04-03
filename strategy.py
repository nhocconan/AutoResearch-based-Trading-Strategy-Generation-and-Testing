#!/usr/bin/env python3
"""
Experiment #109: 4h Donchian(20) breakout + 1d/1w trend filter + volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian breakouts aligned with BOTH daily and weekly HMA trends, with volume confirmation (>1.5x), capture medium-term momentum while avoiding counter-trend trades. Uses discrete sizing (0.25) and ATR stoploss (2.0*ATR). Target: 75-200 total trades over 4 years (19-50/year). Works in bull/bear via dual timeframe trend filter and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_109_4h_donchian20_1d_1w_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = close_1d.ewm(span=half_len, adjust=False).mean()
    wma_full = close_1d.ewm(span=21, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_1d = raw_hma.ewm(span=sqrt_len, adjust=False).mean()
    hma_1d_values = hma_1d.values
    daily_trend = np.where(close_1d > hma_1d_values, 1, -1)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # === HTF: 1w data for HMA(21) trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = pd.Series(df_1w['close'].values)
    wma_half_w = close_1w.ewm(span=half_len, adjust=False).mean()
    wma_full_w = close_1w.ewm(span=21, adjust=False).mean()
    raw_hma_w = 2 * wma_half_w - wma_full_w
    hma_1w = raw_hma_w.ewm(span=sqrt_len, adjust=False).mean()
    hma_1w_values = hma_1w.values
    weekly_trend = np.where(close_1w > hma_1w_values, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(daily_trend_aligned[i]) or
            np.isnan(weekly_trend_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Dual Timeframe Trend: BOTH 1d AND 1w must agree ---
        bullish_trend = (daily_trend_aligned[i] > 0) and (weekly_trend_aligned[i] > 0)
        bearish_trend = (daily_trend_aligned[i] < 0) and (weekly_trend_aligned[i] < 0)
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32h on 4h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND BOTH daily/weekly bullish
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND BOTH daily/weekly bearish
            elif breakout_down and bearish_trend:
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
Experiment #109: 4h Donchian(20) breakout + 1d/1w trend filter + volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian breakouts aligned with BOTH daily and weekly HMA trends, with volume confirmation (>1.5x), capture medium-term momentum while avoiding counter-trend trades. Uses discrete sizing (0.25) and ATR stoploss (2.0*ATR). Target: 75-200 total trades over 4 years (19-50/year). Works in bull/bear via dual timeframe trend filter and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_109_4h_donchian20_1d_1w_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = close_1d.ewm(span=half_len, adjust=False).mean()
    wma_full = close_1d.ewm(span=21, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_1d = raw_hma.ewm(span=sqrt_len, adjust=False).mean()
    hma_1d_values = hma_1d.values
    daily_trend = np.where(close_1d > hma_1d_values, 1, -1)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # === HTF: 1w data for HMA(21) trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = pd.Series(df_1w['close'].values)
    wma_half_w = close_1w.ewm(span=half_len, adjust=False).mean()
    wma_full_w = close_1w.ewm(span=21, adjust=False).mean()
    raw_hma_w = 2 * wma_half_w - wma_full_w
    hma_1w = raw_hma_w.ewm(span=sqrt_len, adjust=False).mean()
    hma_1w_values = hma_1w.values
    weekly_trend = np.where(close_1w > hma_1w_values, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(daily_trend_aligned[i]) or
            np.isnan(weekly_trend_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Dual Timeframe Trend: BOTH 1d AND 1w must agree ---
        bullish_trend = (daily_trend_aligned[i] > 0) and (weekly_trend_aligned[i] > 0)
        bearish_trend = (daily_trend_aligned[i] < 0) and (weekly_trend_aligned[i] < 0)
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32h on 4h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND BOTH daily/weekly bullish
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND BOTH daily/weekly bearish
            elif breakout_down and bearish_trend:
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