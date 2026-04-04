#!/usr/bin/env python3
"""
Experiment #5799: 6h Donchian(20) breakout + 12h Supertrend(ATR=10,mult=3) trend filter + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Supertrend direction capture strong continuation moves while avoiding counter-trend whipsaws. Uses 12h Supertrend for robust trend regime (avoids EMA whipsaws in choppy markets). Volume confirmation filters false breakouts. ATR-based trailing stop manages risk. Targets 75-200 trades over 4 years with discrete sizing 0.25 to minimize fee drag. Works in both bull (breakouts with trend) and bear (breakouts against trend filtered by Supertrend regime).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5799_6h_donchian20_12h_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for Supertrend(ATR=10,mult=3) trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 10:
        # Calculate Supertrend on 12h
        hl2 = (df_12h['high'] + df_12h['low']) / 2
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - np.roll(df_12h['close'], 1))
        tr3 = np.abs(df_12h['low'] - np.roll(df_12h['close'], 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
        
        upper_band = hl2 + 3 * atr_12h
        lower_band = hl2 - 3 * atr_12h
        
        supertrend = np.full(len(hl2), np.nan, dtype=np.float64)
        direction = np.full(len(hl2), 1, dtype=np.float64)  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(hl2)):
            if np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
                supertrend[i] = np.nan
                direction[i] = direction[i-1]
                continue
                
            # Upper band logic
            if upper_band[i] < upper_band[i-1] or df_12h['close'].iloc[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
                
            # Lower band logic
            if lower_band[i] > lower_band[i-1] or df_12h['close'].iloc[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
                
            # Trend direction
            if df_12h['close'].iloc[i] > upper_band[i-1]:
                direction[i] = 1
            elif df_12h['close'].iloc[i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            # Supertrend value
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
    else:
        supertrend = np.full(len(df_12h), np.nan)
        direction = np.zeros(len(df_12h))
    
    # Align 12h Supertrend direction to 6h timeframe (shifted by 1 for completed 12h bars only)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(10) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 10, 10)  # Donchian, volume avg, ATR, Supertrend warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(supertrend_direction_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        # Regime filter: 12h Supertrend direction for trend alignment
        regime_long = supertrend_direction_aligned[i] > 0
        regime_short = supertrend_direction_aligned[i] < 0
        
        # Entry conditions: breakout in direction of 12h Supertrend trend with volume confirmation
        long_setup = breakout_up and regime_long and volume_confirmed
        short_setup = breakout_down and regime_short and volume_confirmed
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
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
Experiment #5799: 6h Donchian(20) breakout + 12h Supertrend(ATR=10,mult=3) trend filter + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Supertrend direction capture strong continuation moves while avoiding counter-trend whipsaws. Uses 12h Supertrend for robust trend regime (avoids EMA whipsaws in choppy markets). Volume confirmation filters false breakouts. ATR-based trailing stop manages risk. Targets 75-200 trades over 4 years with discrete sizing 0.25 to minimize fee drag. Works in both bull (breakouts with trend) and bear (breakouts against trend filtered by Supertrend regime).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5799_6h_donchian20_12h_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for Supertrend(ATR=10,mult=3) trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 10:
        # Calculate Supertrend on 12h
        hl2 = (df_12h['high'] + df_12h['low']) / 2
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - np.roll(df_12h['close'], 1))
        tr3 = np.abs(df_12h['low'] - np.roll(df_12h['close'], 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
        
        upper_band = hl2 + 3 * atr_12h
        lower_band = hl2 - 3 * atr_12h
        
        supertrend = np.full(len(hl2), np.nan, dtype=np.float64)
        direction = np.full(len(hl2), 1, dtype=np.float64)  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(hl2)):
            if np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
                supertrend[i] = np.nan
                direction[i] = direction[i-1]
                continue
                
            # Upper band logic
            if upper_band[i] < upper_band[i-1] or df_12h['close'].iloc[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
                
            # Lower band logic
            if lower_band[i] > lower_band[i-1] or df_12h['close'].iloc[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
                
            # Trend direction
            if df_12h['close'].iloc[i] > upper_band[i-1]:
                direction[i] = 1
            elif df_12h['close'].iloc[i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            # Supertrend value
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
    else:
        supertrend = np.full(len(df_12h), np.nan)
        direction = np.zeros(len(df_12h))
    
    # Align 12h Supertrend direction to 6h timeframe (shifted by 1 for completed 12h bars only)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(10) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 10, 10)  # Donchian, volume avg, ATR, Supertrend warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(supertrend_direction_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        # Regime filter: 12h Supertrend direction for trend alignment
        regime_long = supertrend_direction_aligned[i] > 0
        regime_short = supertrend_direction_aligned[i] < 0
        
        # Entry conditions: breakout in direction of 12h Supertrend trend with volume confirmation
        long_setup = breakout_up and regime_long and volume_confirmed
        short_setup = breakout_down and regime_short and volume_confirmed
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals