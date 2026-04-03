#!/usr/bin/env python3
"""
Experiment #078: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Daily Donchian breakouts above 20-period high or below 20-period low, confirmed by 
weekly HMA(21) trend direction and volume >1.5x average, capture strong momentum moves. 
In bull markets, breakouts above upper band with weekly HMA up yield longs; in bear markets,
breakouts below lower band with weekly HMA down yield shorts. Volume filter reduces false breakouts.
ATR(14) stoploss limits drawdown. Target: 30-100 trades over 4 years on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_078_1d_donchian20_1w_hma_vol_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA(21) ===
    def calculate_hma(arr, period):
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))
        
        def wma(values, window):
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        
        wma_half = np.full_like(arr, np.nan)
        wma_full = np.full_like(arr, np.nan)
        
        for i in range(half_period, len(arr)):
            wma_half[i] = wma(arr[i - half_period + 1:i + 1], half_period)[-1]
        for i in range(period, len(arr)):
            wma_full[i] = wma(arr[i - period + 1:i + 1], period)[-1]
        
        raw_hma = 2 * wma_half - wma_full
        hma = np.full_like(arr, np.nan)
        
        for i in range(sqrt_period - 1, len(arr)):
            start_idx = i - sqrt_period + 1
            if start_idx >= 0 and not np.isnan(raw_hma[i]):
                hma[i] = wma(raw_hma[start_idx:i + 1], sqrt_period)[-1]
        
        return hma
    
    # Calculate HMA on 1w close
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # === 1d Indicators: Donchian Channel(20) ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 1d Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Warmup for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume confirmation threshold
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                if low[i] <= highest_since_entry - 2.5 * atr[i]:  # 2.5 ATR stoploss
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                if high[i] >= lowest_since_entry + 2.5 * atr[i]:  # 2.5 ATR stoploss
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: price breaks above Donchian upper + weekly HMA up + volume spike
        if (price > donchian_upper[i-1] and 
            hma_1w_aligned[i] > hma_1w_aligned[i-1] and 
            vol_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        
        # Short: price breaks below Donchian lower + weekly HMA down + volume spike
        elif (price < donchian_lower[i-1] and 
              hma_1w_aligned[i] < hma_1w_aligned[i-1] and 
              vol_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals