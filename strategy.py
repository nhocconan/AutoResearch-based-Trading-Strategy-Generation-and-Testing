#!/usr/bin/env python3
"""
Experiment #1542: 12h Donchian(20) Breakout + 1d Trend + Volume Confirmation + Chop Filter
HYPOTHESIS: 12h Donchian breakouts with 1d EMA trend alignment, volume spike (>1.8x), and choppiness regime filter (CHOP > 61.8 = ranging) capture medium-term swings while minimizing trades. Only enter in trending markets (CHOP < 61.8) with volume confirmation. Fixed size 0.25 balances drawdown and opportunity. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1542_12h_donchian20_1d_trend_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 12h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index: higher = more ranging, lower = more trending"""
        atr_sum = np.zeros_like(close_arr)
        true_range = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            true_range[i] = max(high[i] - low[i], abs(high[i] - close_arr[i-1]), abs(low[i] - close_arr[i-1]))
        true_range[0] = high[0] - low[0]
        atr_sum = pd.Series(true_range).rolling(window=window, min_periods=window).sum().values
        max_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        chop = np.zeros_like(close_arr)
        for i in range(window-1, len(close_arr)):
            if atr_sum[i] > 0 and max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(window)
            else:
                chop[i] = 50.0  # neutral when undefined
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    chop_filter = chop < 61.8  # True when trending (CHOP < 61.8), False when ranging
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(50) and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 12h)
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
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
        # Require 1d trend alignment
        trend_following = trend_1d_aligned[i] != 0  # Always true unless NaN
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        in_trending_regime = chop_filter[i]
        
        if trend_following and volume_spike and in_trending_regime:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and trend_1d_aligned[i] > 0:  # Uptrend breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and trend_1d_aligned[i] < 0:  # Downtrend breakdown
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
Experiment #1542: 12h Donchian(20) Breakout + 1d Trend + Volume Confirmation + Chop Filter
HYPOTHESIS: 12h Donchian breakouts with 1d EMA trend alignment, volume spike (>1.8x), and choppiness regime filter (CHOP > 61.8 = ranging) capture medium-term swings while minimizing trades. Only enter in trending markets (CHOP < 61.8) with volume confirmation. Fixed size 0.25 balances drawdown and opportunity. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1542_12h_donchian20_1d_trend_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 12h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index: higher = more ranging, lower = more trending"""
        atr_sum = np.zeros_like(close_arr)
        true_range = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            true_range[i] = max(high[i] - low[i], abs(high[i] - close_arr[i-1]), abs(low[i] - close_arr[i-1]))
        true_range[0] = high[0] - low[0]
        atr_sum = pd.Series(true_range).rolling(window=window, min_periods=window).sum().values
        max_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        chop = np.zeros_like(close_arr)
        for i in range(window-1, len(close_arr)):
            if atr_sum[i] > 0 and max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(window)
            else:
                chop[i] = 50.0  # neutral when undefined
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    chop_filter = chop < 61.8  # True when trending (CHOP < 61.8), False when ranging
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(50) and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 12h)
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
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
        # Require 1d trend alignment
        trend_following = trend_1d_aligned[i] != 0  # Always true unless NaN
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        in_trending_regime = chop_filter[i]
        
        if trend_following and volume_spike and in_trending_regime:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and trend_1d_aligned[i] > 0:  # Uptrend breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and trend_1d_aligned[i] < 0:  # Downtrend breakdown
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