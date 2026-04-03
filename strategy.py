#!/usr/bin/env python3
"""
Experiment #254: 1h Donchian(20) Breakout + 4h/1d HTF Trend + Volume Spike + Session Filter

HYPOTHESIS: 1h Donchian breakouts aligned with 4h/1d EMA50 trend capture momentum with proper timing.
Volume confirmation (2.0x average) ensures institutional participation. Session filter (08-20 UTC) reduces noise.
Target: 15-37 trades/year on 1h timeframe to minimize fee drag while capturing significant moves.
Uses discrete position sizing (0.20) and ATR-based trailing stop (2.5x) for risk control.
Works in both bull and bear markets by only taking breakouts in the direction of the HTF trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_htf_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h and 1d data for EMA50 trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
        htf_4h_trend = np.where(close[:len(ema_4h_aligned)] > ema_4h_aligned, 1, -1)
    else:
        htf_4h_trend = np.full(n, 0)
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
        htf_1d_trend = np.where(close[:len(ema_1d_aligned)] > ema_1d_aligned, 1, -1)
    else:
        htf_1d_trend = np.full(n, 0)
    
    # Combined HTF trend: require both 4h and 1d to agree
    htf_trend = np.where((htf_4h_trend > 0) & (htf_1d_trend > 0), 1,
                         np.where((htf_4h_trend < 0) & (htf_1d_trend < 0), -1, 0))
    
    # === 1h Indicators ===
    # ATR(14) for stoploss and volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or i >= len(htf_trend)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Trend Filter from HTF EMA50 ---
        trend_ok_long = htf_trend[i] > 0   # HTF trend bullish
        trend_ok_short = htf_trend[i] < 0  # HTF trend bearish
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~3h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR HTF trend turns bearish
                    if close[i] <= dc_lower_20[i] or htf_trend[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR HTF trend turns bullish
                    if close[i] >= dc_upper_20[i] or htf_trend[i] > 0:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with volume confirmation and bullish HTF trend
        if bullish_breakout and vol_ok and trend_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish HTF trend
        elif bearish_breakout and vol_ok and trend_ok_short:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #254: 1h Donchian(20) Breakout + 4h/1d HTF Trend + Volume Spike + Session Filter

HYPOTHESIS: 1h Donchian breakouts aligned with 4h/1d EMA50 trend capture momentum with proper timing.
Volume confirmation (2.0x average) ensures institutional participation. Session filter (08-20 UTC) reduces noise.
Target: 15-37 trades/year on 1h timeframe to minimize fee drag while capturing significant moves.
Uses discrete position sizing (0.20) and ATR-based trailing stop (2.5x) for risk control.
Works in both bull and bear markets by only taking breakouts in the direction of the HTF trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_htf_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h and 1d data for EMA50 trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
        htf_4h_trend = np.where(close[:len(ema_4h_aligned)] > ema_4h_aligned, 1, -1)
    else:
        htf_4h_trend = np.full(n, 0)
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
        htf_1d_trend = np.where(close[:len(ema_1d_aligned)] > ema_1d_aligned, 1, -1)
    else:
        htf_1d_trend = np.full(n, 0)
    
    # Combined HTF trend: require both 4h and 1d to agree
    htf_trend = np.where((htf_4h_trend > 0) & (htf_1d_trend > 0), 1,
                         np.where((htf_4h_trend < 0) & (htf_1d_trend < 0), -1, 0))
    
    # === 1h Indicators ===
    # ATR(14) for stoploss and volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or i >= len(htf_trend)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Trend Filter from HTF EMA50 ---
        trend_ok_long = htf_trend[i] > 0   # HTF trend bullish
        trend_ok_short = htf_trend[i] < 0  # HTF trend bearish
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~3h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR HTF trend turns bearish
                    if close[i] <= dc_lower_20[i] or htf_trend[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR HTF trend turns bullish
                    if close[i] >= dc_upper_20[i] or htf_trend[i] > 0:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with volume confirmation and bullish HTF trend
        if bullish_breakout and vol_ok and trend_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish HTF trend
        elif bearish_breakout and vol_ok and trend_ok_short:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>