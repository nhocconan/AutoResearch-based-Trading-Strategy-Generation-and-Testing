#!/usr/bin/env python3
"""
Experiment #6249: 4h Donchian(20) breakout + 1d/1w EMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 1d and 1w EMA trend direction capture strong momentum moves. Volume >1.8x average confirms institutional participation. Uses discrete sizing (0.25) and ATR trailing stop (2.5x) to manage risk and fee drag. Target: 75-200 trades over 4 years (19-50/year) for 4h timeframe.
Works in bull markets via breakout continuation and in bear markets via faded breakouts at extreme levels (counter-trend entries when price deviates far from 1w EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6249_4h_donchian20_1d_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for EMA trend (stronger filter) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50) + 1  # Donchian, volume avg, ATR, EMA + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
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
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter for stronger signals
        
        # Trend filters: 
        # Long: price above both 1d and 1w EMA (strong uptrend)
        # Short: price below both 1d and 1w EMA (strong downtrend)
        # Counter-trend fade: price far from 1w EMA (>2.5*ATR) in opposite direction
        ema_1d_val = ema_1d_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        
        # Strong trend continuation entries
        long_trend = breakout_up and volume_confirmed and (price > ema_1d_val) and (price > ema_1w_val)
        short_trend = breakout_down and volume_confirmed and (price < ema_1d_val) and (price < ema_1w_val)
        
        # Counter-trend fade entries (only when price is extremely extended)
        long_fade = breakout_down and volume_confirmed and (price < ema_1w_val - 2.5 * atr[i])
        short_fade = breakout_up and volume_confirmed and (price > ema_1w_val + 2.5 * atr[i])
        
        long_entry = long_trend or long_fade
        short_entry = short_trend or short_fade
        
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
Experiment #6249: 4h Donchian(20) breakout + 1d/1w EMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 1d and 1w EMA trend direction capture strong momentum moves. Volume >1.8x average confirms institutional participation. Uses discrete sizing (0.25) and ATR trailing stop (2.5x) to manage risk and fee drag. Target: 75-200 trades over 4 years (19-50/year) for 4h timeframe.
Works in bull markets via breakout continuation and in bear markets via faded breakouts at extreme levels (counter-trend entries when price deviates far from 1w EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6249_4h_donchian20_1d_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for EMA trend (stronger filter) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50) + 1  # Donchian, volume avg, ATR, EMA + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
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
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter for stronger signals
        
        # Trend filters: 
        # Long: price above both 1d and 1w EMA (strong uptrend)
        # Short: price below both 1d and 1w EMA (strong downtrend)
        # Counter-trend fade: price far from 1w EMA (>2.5*ATR) in opposite direction
        ema_1d_val = ema_1d_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        
        # Strong trend continuation entries
        long_trend = breakout_up and volume_confirmed and (price > ema_1d_val) and (price > ema_1w_val)
        short_trend = breakout_down and volume_confirmed and (price < ema_1d_val) and (price < ema_1w_val)
        
        # Counter-trend fade entries (only when price is extremely extended)
        long_fade = breakout_down and volume_confirmed and (price < ema_1w_val - 2.5 * atr[i])
        short_fade = breakout_up and volume_confirmed and (price > ema_1w_val + 2.5 * atr[i])
        
        long_entry = long_trend or long_fade
        short_entry = short_trend or short_fade
        
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