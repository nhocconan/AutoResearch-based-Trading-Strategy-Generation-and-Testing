#!/usr/bin/env python3
"""
Experiment #4475: 6h Donchian(20) Breakout + 1w Camarilla Pivot + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly Camarilla pivot structure (R3/S3 for mean reversion, R4/S4 for breakout) and confirmed by volume (>2.0x average) capture institutional moves with minimal false signals. Weekly pivot provides structural bias from higher timeframe, reducing whipsaws in both bull and bear markets. Volume filters low-conviction moves. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4475_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1w OHLC for Camarilla Pivot ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly Camarilla levels from previous week's OHLC
        H_1w = df_1w['high'].values
        L_1w = df_1w['low'].values
        C_1w = df_1w['close'].values
        
        # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
        R4_1w = C_1w + ((H_1w - L_1w) * 1.1 / 2)
        R3_1w = C_1w + ((H_1w - L_1w) * 1.1 / 4)
        S3_1w = C_1w - ((H_1w - L_1w) * 1.1 / 4)
        S4_1w = C_1w - ((H_1w - L_1w) * 1.1 / 2)
        
        # Align to 6h timeframe (shifted by 1 week for completed bars only)
        R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
        R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
        S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
        S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    else:
        R4_1w_aligned = np.full(n, np.nan)
        R3_1w_aligned = np.full(n, np.nan)
        S3_1w_aligned = np.full(n, np.nan)
        S4_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(R4_1w_aligned[i]) or np.isnan(R3_1w_aligned[i]) or
            np.isnan(S3_1w_aligned[i]) or np.isnan(S4_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Weekly Camarilla logic:
        # - Price between R3 and S3: mean reversion zone (fade extremes)
        # - Price > R4 or < S4: breakout zone (continuation)
        in_mid_zone = (price > S3_1w_aligned[i]) and (price < R3_1w_aligned[i])
        in_breakout_zone = (price >= R4_1w_aligned[i]) or (price <= S4_1w_aligned[i])
        
        # Long conditions:
        # 1. Breakout zone: upward Donchian breakout + continuation bias
        # 2. Mean reversion zone: price near S3 + upward Donchian breakout (bounce)
        long_breakout = breakout_up and in_breakout_zone and (price > R4_1w_aligned[i])
        long_reversion = breakout_up and in_mid_zone and (price < S3_1w_aligned[i] + (R3_1w_aligned[i] - S3_1w_aligned[i]) * 0.3)
        long_entry = long_breakout or long_reversion
        
        # Short conditions:
        # 1. Breakout zone: downward Donchian breakout + continuation bias
        # 2. Mean reversion zone: price near R3 + downward Donchian breakout (rejection)
        short_breakout = breakout_down and in_breakout_zone and (price < S4_1w_aligned[i])
        short_reversion = breakout_down and in_mid_zone and (price > R3_1w_aligned[i] - (R3_1w_aligned[i] - S3_1w_aligned[i]) * 0.3)
        short_entry = short_breakout or short_reversion
        
        # Require volume confirmation
        long_entry = long_entry and volume_confirm
        short_entry = short_entry and volume_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
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
Experiment #4475: 6h Donchian(20) Breakout + 1w Camarilla Pivot + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly Camarilla pivot structure (R3/S3 for mean reversion, R4/S4 for breakout) and confirmed by volume (>2.0x average) capture institutional moves with minimal false signals. Weekly pivot provides structural bias from higher timeframe, reducing whipsaws in both bull and bear markets. Volume filters low-conviction moves. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4475_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1w OHLC for Camarilla Pivot ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly Camarilla levels from previous week's OHLC
        H_1w = df_1w['high'].values
        L_1w = df_1w['low'].values
        C_1w = df_1w['close'].values
        
        # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
        R4_1w = C_1w + ((H_1w - L_1w) * 1.1 / 2)
        R3_1w = C_1w + ((H_1w - L_1w) * 1.1 / 4)
        S3_1w = C_1w - ((H_1w - L_1w) * 1.1 / 4)
        S4_1w = C_1w - ((H_1w - L_1w) * 1.1 / 2)
        
        # Align to 6h timeframe (shifted by 1 week for completed bars only)
        R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
        R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
        S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
        S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    else:
        R4_1w_aligned = np.full(n, np.nan)
        R3_1w_aligned = np.full(n, np.nan)
        S3_1w_aligned = np.full(n, np.nan)
        S4_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(R4_1w_aligned[i]) or np.isnan(R3_1w_aligned[i]) or
            np.isnan(S3_1w_aligned[i]) or np.isnan(S4_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Weekly Camarilla logic:
        # - Price between R3 and S3: mean reversion zone (fade extremes)
        # - Price > R4 or < S4: breakout zone (continuation)
        in_mid_zone = (price > S3_1w_aligned[i]) and (price < R3_1w_aligned[i])
        in_breakout_zone = (price >= R4_1w_aligned[i]) or (price <= S4_1w_aligned[i])
        
        # Long conditions:
        # 1. Breakout zone: upward Donchian breakout + continuation bias
        # 2. Mean reversion zone: price near S3 + upward Donchian breakout (bounce)
        long_breakout = breakout_up and in_breakout_zone and (price > R4_1w_aligned[i])
        long_reversion = breakout_up and in_mid_zone and (price < S3_1w_aligned[i] + (R3_1w_aligned[i] - S3_1w_aligned[i]) * 0.3)
        long_entry = long_breakout or long_reversion
        
        # Short conditions:
        # 1. Breakout zone: downward Donchian breakout + continuation bias
        # 2. Mean reversion zone: price near R3 + downward Donchian breakout (rejection)
        short_breakout = breakout_down and in_breakout_zone and (price < S4_1w_aligned[i])
        short_reversion = breakout_down and in_mid_zone and (price > R3_1w_aligned[i] - (R3_1w_aligned[i] - S3_1w_aligned[i]) * 0.3)
        short_entry = short_breakout or short_reversion
        
        # Require volume confirmation
        long_entry = long_entry and volume_confirm
        short_entry = short_entry and volume_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals