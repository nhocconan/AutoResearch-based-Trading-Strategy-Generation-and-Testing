#!/usr/bin/env python3
"""
Experiment #434: 1h Donchian Breakout + 4h Volume Spike + 1d Trend Filter

HYPOTHESIS: Donchian channel breakouts on 1h timeframe, confirmed by 4h volume spike (>2.0x average) 
and aligned with 1d trend (price > EMA50 for long, < EMA50 for short), captures high-probability 
trend continuation moves. Using 4h/1d for signal direction and 1h for precise entry timing reduces 
false breakouts. Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag 
while maintaining edge in both bull and bear markets via volume confirmation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for volume spike confirmation (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate volume ratio (current vs 20-period average) on 4h
    if len(df_4h) >= 20:
        vol_4h = df_4h['volume'].values
        vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_4h = np.zeros(len(vol_4h))
        vol_ratio_4h[20:] = vol_4h[20:] / vol_ma_20[20:]
        vol_ratio_4h[:20] = 1.0  # Neutral for warmup
        vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    else:
        vol_ratio_4h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Session Filter: 08-20 UTC ===
    # open_time is already datetime64[ms], access via index
    hours = prices.index.hour  # Pre-compute once before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 50)  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (price > 1d EMA50 for long, < for short) ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) on 4h ---
        volume_spike = vol_ratio_4h_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian upper band (trailing)
                if close[i] >= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian lower band (trailing)
                if close[i] <= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume spike and uptrend
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above upper band
            volume_spike and                 # 4h volume confirmation
            price_above_1d_ema               # 1d uptrend filter
        )
        
        # Short: Price breaks below Donchian lower band with volume spike and downtrend
        short_condition = (
            close[i] < lowest_low[i] and     # Breakdown below lower band
            volume_spike and                 # 4h volume confirmation
            price_below_1d_ema               # 1d downtrend filter
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #434: 1h Donchian Breakout + 4h Volume Spike + 1d Trend Filter

HYPOTHESIS: Donchian channel breakouts on 1h timeframe, confirmed by 4h volume spike (>2.0x average) 
and aligned with 1d trend (price > EMA50 for long, < EMA50 for short), captures high-probability 
trend continuation moves. Using 4h/1d for signal direction and 1h for precise entry timing reduces 
false breakouts. Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag 
while maintaining edge in both bull and bear markets via volume confirmation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for volume spike confirmation (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate volume ratio (current vs 20-period average) on 4h
    if len(df_4h) >= 20:
        vol_4h = df_4h['volume'].values
        vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_4h = np.zeros(len(vol_4h))
        vol_ratio_4h[20:] = vol_4h[20:] / vol_ma_20[20:]
        vol_ratio_4h[:20] = 1.0  # Neutral for warmup
        vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    else:
        vol_ratio_4h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Session Filter: 08-20 UTC ===
    # open_time is already datetime64[ms], access via index
    hours = prices.index.hour  # Pre-compute once before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 50)  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (price > 1d EMA50 for long, < for short) ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) on 4h ---
        volume_spike = vol_ratio_4h_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian upper band (trailing)
                if close[i] >= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian lower band (trailing)
                if close[i] <= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume spike and uptrend
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above upper band
            volume_spike and                 # 4h volume confirmation
            price_above_1d_ema               # 1d uptrend filter
        )
        
        # Short: Price breaks below Donchian lower band with volume spike and downtrend
        short_condition = (
            close[i] < lowest_low[i] and     # Breakdown below lower band
            volume_spike and                 # 4h volume confirmation
            price_below_1d_ema               # 1d downtrend filter
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals