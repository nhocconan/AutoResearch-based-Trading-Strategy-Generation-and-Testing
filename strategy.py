#!/usr/bin/env python3
"""
Experiment #1870: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Daily Donchian breakouts capture major trend moves. Weekly HMA filter ensures alignment with primary trend, reducing false breakouts. Volume confirmation (>1.5x average) adds conviction. ATR-based stoploss manages risk. Works in bull markets via upside breakouts and bear markets via downside breakouts. Target: 30-100 total trades over 4 years (7-25/year) with position sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1870_1d_donchian20_1w_hma_vol_v1"
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
    
    # Weekly HMA(21) for trend direction
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt_period = int(np.sqrt(period))
        if half < 1 or sqrt_period < 1:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        # Pad to original length
        wma_half_padded = np.concatenate([np.full(period - half, np.nan), wma_half])
        wma_full_padded = np.concatenate([np.full(period - len(wma_full), np.nan), wma_full])
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_values = wma(raw_hma, sqrt_period)
        # Final padding
        hma_values = np.concatenate([np.full(len(arr) - len(hma_values), np.nan), hma_values])
        return hma_values
    
    hma_21_1w = hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1d Indicators: Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    warmup = max(lookback, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or trend change ---
        if in_position:
            # Check stoploss
            if position_side > 0:  # Long position
                if price <= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price >= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Exit if weekly trend flips
            if (position_side > 0 and trend_1w_aligned[i] < 0) or \
               (position_side < 0 and trend_1w_aligned[i] > 0):
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly trend alignment for bias
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Breakout conditions
            if trend_bias > 0 and price > highest_high[i]:
                # Long breakout above upper Donchian
                in_position = True
                position_side = 1
                entry_price = price
                stoploss_price = price - 2.5 * atr[i]  # 2.5 ATR stoploss
                signals[i] = SIZE
            elif trend_bias < 0 and price < lowest_low[i]:
                # Short breakout below lower Donchian
                in_position = True
                position_side = -1
                entry_price = price
                stoploss_price = price + 2.5 * atr[i]  # 2.5 ATR stoploss
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1870: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Daily Donchian breakouts capture major trend moves. Weekly HMA filter ensures alignment with primary trend, reducing false breakouts. Volume confirmation (>1.5x average) adds conviction. ATR-based stoploss manages risk. Works in bull markets via upside breakouts and bear markets via downside breakouts. Target: 30-100 total trades over 4 years (7-25/year) with position sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1870_1d_donchian20_1w_hma_vol_v1"
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
    
    # Weekly HMA(21) for trend direction
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt_period = int(np.sqrt(period))
        if half < 1 or sqrt_period < 1:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        # Pad to original length
        wma_half_padded = np.concatenate([np.full(period - half, np.nan), wma_half])
        wma_full_padded = np.concatenate([np.full(period - len(wma_full), np.nan), wma_full])
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_values = wma(raw_hma, sqrt_period)
        # Final padding
        hma_values = np.concatenate([np.full(len(arr) - len(hma_values), np.nan), hma_values])
        return hma_values
    
    hma_21_1w = hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1d Indicators: Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    warmup = max(lookback, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or trend change ---
        if in_position:
            # Check stoploss
            if position_side > 0:  # Long position
                if price <= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price >= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Exit if weekly trend flips
            if (position_side > 0 and trend_1w_aligned[i] < 0) or \
               (position_side < 0 and trend_1w_aligned[i] > 0):
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly trend alignment for bias
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Breakout conditions
            if trend_bias > 0 and price > highest_high[i]:
                # Long breakout above upper Donchian
                in_position = True
                position_side = 1
                entry_price = price
                stoploss_price = price - 2.5 * atr[i]  # 2.5 ATR stoploss
                signals[i] = SIZE
            elif trend_bias < 0 and price < lowest_low[i]:
                # Short breakout below lower Donchian
                in_position = True
                position_side = -1
                entry_price = price
                stoploss_price = price + 2.5 * atr[i]  # 2.5 ATR stoploss
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals