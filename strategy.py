#!/usr/bin/env python3
"""
Experiment #1827: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R1/S1, R2/S2) and volume confirmation (>1.3x average) capture medium-term swings in both bull and bear markets. Weekly pivot levels act as dynamic support/resistance from higher timeframe structure. Position size fixed at 0.25. Target: 75-150 total trades over 4 years (19-37/year) by using tight entry conditions and multi-timeframe confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1827_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Ensure we have enough data for weekly calculation
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly high, low, close from daily data
    # Group by week (starting Monday) - using 5-day approximation for weekly
    # In practice, we'll use rolling window of 5 days for weekly pivot
    dh_1d = df_1d['high'].values
    dl_1d = df_1d['low'].values
    dc_1d = df_1d['close'].values
    
    # Weekly pivot: use 5-day rolling window (approx 1 week)
    # Weekly high = max(high) over 5 days
    # Weekly low = min(low) over 5 days  
    # Weekly close = close of last day in period
    week_high = pd.Series(dh_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(dl_1d).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(dc_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values
    
    # Calculate pivot points
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    r3 = week_high + 2 * (pivot - week_low)
    s3 = week_low - 2 * (week_high - pivot)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Determine market regime relative to weekly pivot
        # In uptrend: price above pivot, look for longs at support, shorts at resistance
        # In downtrend: price below pivot, look for shorts at resistance, longs at support
        
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            # Price above pivot = bullish bias
            if price > pivot_aligned[i]:
                # Look for long opportunities: break above R1 or pullback to S1
                if price > donch_high[i] and price > r1_aligned[i]:  # Break above R1 with Donchian breakout
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < donch_low[i] and price > s1_aligned[i] and price < pivot_aligned[i]:  # Pullback to S1 in uptrend
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            # Price below pivot = bearish bias  
            else:
                # Look for short opportunities: break below S1 or pullback to R1
                if price < donch_low[i] and price < s1_aligned[i]:  # Break below S1 with Donchian breakdown
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif price > donch_high[i] and price < r1_aligned[i] and price > pivot_aligned[i]:  # Pullback to R1 in downtrend
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
Experiment #1827: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R1/S1, R2/S2) and volume confirmation (>1.3x average) capture medium-term swings in both bull and bear markets. Weekly pivot levels act as dynamic support/resistance from higher timeframe structure. Position size fixed at 0.25. Target: 75-150 total trades over 4 years (19-37/year) by using tight entry conditions and multi-timeframe confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1827_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Ensure we have enough data for weekly calculation
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly high, low, close from daily data
    # Group by week (starting Monday) - using 5-day approximation for weekly
    # In practice, we'll use rolling window of 5 days for weekly pivot
    dh_1d = df_1d['high'].values
    dl_1d = df_1d['low'].values
    dc_1d = df_1d['close'].values
    
    # Weekly pivot: use 5-day rolling window (approx 1 week)
    # Weekly high = max(high) over 5 days
    # Weekly low = min(low) over 5 days  
    # Weekly close = close of last day in period
    week_high = pd.Series(dh_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(dl_1d).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(dc_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values
    
    # Calculate pivot points
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    r3 = week_high + 2 * (pivot - week_low)
    s3 = week_low - 2 * (week_high - pivot)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Determine market regime relative to weekly pivot
        # In uptrend: price above pivot, look for longs at support, shorts at resistance
        # In downtrend: price below pivot, look for shorts at resistance, longs at support
        
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            # Price above pivot = bullish bias
            if price > pivot_aligned[i]:
                # Look for long opportunities: break above R1 or pullback to S1
                if price > donch_high[i] and price > r1_aligned[i]:  # Break above R1 with Donchian breakout
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < donch_low[i] and price > s1_aligned[i] and price < pivot_aligned[i]:  # Pullback to S1 in uptrend
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            # Price below pivot = bearish bias  
            else:
                # Look for short opportunities: break below S1 or pullback to R1
                if price < donch_low[i] and price < s1_aligned[i]:  # Break below S1 with Donchian breakdown
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif price > donch_high[i] and price < r1_aligned[i] and price > pivot_aligned[i]:  # Pullback to R1 in downtrend
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