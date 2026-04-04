#!/usr/bin/env python3
"""
Experiment #5024: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 1d timeframe, Donchian(20) breakouts in direction of 1w HMA21 trend with volume confirmation (>2x average) capture strong momentum moves. Uses ATR(14) trailing stop (2.5x) to limit downside. Designed for 7-25 trades/year on 1d timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5024_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA21 for trend filter ===
    if len(df_1w) >= 21:
        # Hull Moving Average calculation
        half_len = len(df_1w) // 2
        sqrt_len = int(np.sqrt(len(df_1w)))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_1w = df_1w['close'].values
        wma_half = np.array([wma(close_1w[i:i+half_len], half_len)[-1] 
                            if i+half_len <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        wma_full = np.array([wma(close_1w[i:i+len(close_1w)], len(close_1w))[-1] 
                            if i+len(close_1w) <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        wma_sqrt = np.array([wma(close_1w[i:i+sqrt_len], sqrt_len)[-1] 
                            if i+sqrt_len <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_raw = 2 * wma_half - wma_full
        hma_1w = np.array([wma(hma_raw[i:i+sqrt_len], sqrt_len)[-1] 
                          if i+sqrt_len <= len(hma_raw) else np.nan 
                          for i in range(len(hma_raw))])
    else:
        hma_1w = np.full(len(df_1w), np.nan)
    
    # Align HTF HMA21 to 1d timeframe
    if len(hma_1w) > 0:
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with trend alignment
        breakout_long = (price >= high_roll[i]) and (price > hma_1w_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_1w_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #5024: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 1d timeframe, Donchian(20) breakouts in direction of 1w HMA21 trend with volume confirmation (>2x average) capture strong momentum moves. Uses ATR(14) trailing stop (2.5x) to limit downside. Designed for 7-25 trades/year on 1d timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5024_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA21 for trend filter ===
    if len(df_1w) >= 21:
        # Hull Moving Average calculation
        half_len = len(df_1w) // 2
        sqrt_len = int(np.sqrt(len(df_1w)))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_1w = df_1w['close'].values
        wma_half = np.array([wma(close_1w[i:i+half_len], half_len)[-1] 
                            if i+half_len <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        wma_full = np.array([wma(close_1w[i:i+len(close_1w)], len(close_1w))[-1] 
                            if i+len(close_1w) <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        wma_sqrt = np.array([wma(close_1w[i:i+sqrt_len], sqrt_len)[-1] 
                            if i+sqrt_len <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_raw = 2 * wma_half - wma_full
        hma_1w = np.array([wma(hma_raw[i:i+sqrt_len], sqrt_len)[-1] 
                          if i+sqrt_len <= len(hma_raw) else np.nan 
                          for i in range(len(hma_raw))])
    else:
        hma_1w = np.full(len(df_1w), np.nan)
    
    # Align HTF HMA21 to 1d timeframe
    if len(hma_1w) > 0:
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with trend alignment
        breakout_long = (price >= high_roll[i]) and (price > hma_1w_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_1w_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals