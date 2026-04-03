#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 1d timeframe, filtered by weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) and 1w volume confirmation (> 1.5x average), creates a robust strategy that captures strong momentum moves in both bull and bear markets. Weekly pivot provides institutional reference points, Donchian breakouts capture breakout momentum, and volume confirms participation. Targets 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag while capturing high-probability trend continuations. Designed to work in both bull and bear regimes by using pivot direction as adaptive bias and requiring volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    # Reuse df_1w for pivot calculation
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        pivot_1w = (high_1w + low_1w + close_1w) / 3.0
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    else:
        pivot_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Calculate Donchian channels (20-period) on 1d
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        window_high = high[i-lookback+1:i+1]
        window_low = low[i-lookback+1:i+1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1w_aligned[i]) or np.isnan(pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Directional Filter: Weekly pivot bias ---
        bullish_bias = close[i] > pivot_1w_aligned[i]   # Price above weekly pivot = bullish
        bearish_bias = close[i] < pivot_1w_aligned[i]   # Price below weekly pivot = bearish
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.5
        
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
                # Take profit at Donchian upper band (trailing breakout)
                if close[i] <= highest_high[i]:
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
                # Take profit at Donchian lower band (trailing breakout)
                if close[i] >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above highest high with bullish bias and volume
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above Donchian upper band
            bullish_bias and                # Weekly pivot bullish bias
            volume_spike                    # Volume confirmation
        )
        
        # Short: Donchian breakdown below lowest low with bearish bias and volume
        short_condition = (
            close[i] < lowest_low[i] and    # Breakdown below Donchian lower band
            bearish_bias and                # Weekly pivot bearish bias
            volume_spike                    # Volume confirmation
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
Experiment #004: 1d Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 1d timeframe, filtered by weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) and 1w volume confirmation (> 1.5x average), creates a robust strategy that captures strong momentum moves in both bull and bear markets. Weekly pivot provides institutional reference points, Donchian breakouts capture breakout momentum, and volume confirms participation. Targets 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag while capturing high-probability trend continuations. Designed to work in both bull and bear regimes by using pivot direction as adaptive bias and requiring volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    # Reuse df_1w for pivot calculation
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        pivot_1w = (high_1w + low_1w + close_1w) / 3.0
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    else:
        pivot_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Calculate Donchian channels (20-period) on 1d
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        window_high = high[i-lookback+1:i+1]
        window_low = low[i-lookback+1:i+1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1w_aligned[i]) or np.isnan(pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Directional Filter: Weekly pivot bias ---
        bullish_bias = close[i] > pivot_1w_aligned[i]   # Price above weekly pivot = bullish
        bearish_bias = close[i] < pivot_1w_aligned[i]   # Price below weekly pivot = bearish
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.5
        
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
                # Take profit at Donchian upper band (trailing breakout)
                if close[i] <= highest_high[i]:
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
                # Take profit at Donchian lower band (trailing breakout)
                if close[i] >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above highest high with bullish bias and volume
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above Donchian upper band
            bullish_bias and                # Weekly pivot bullish bias
            volume_spike                    # Volume confirmation
        )
        
        # Short: Donchian breakdown below lowest low with bearish bias and volume
        short_condition = (
            close[i] < lowest_low[i] and    # Breakdown below Donchian lower band
            bearish_bias and                # Weekly pivot bearish bias
            volume_spike                    # Volume confirmation
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