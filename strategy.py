#!/usr/bin/env python3
"""
Experiment #001: 4h Donchian(20) breakout + 1d EMA trend + Volume spike

HYPOTHESIS: In both bull and bear markets, strong directional moves begin with price breaking
out of the recent 4h Donchian channel (20-bar high/low). By requiring alignment with the
1d EMA trend (primary trend filter) and a volume spike (confirmation of participation),
we capture high-probability breakouts while avoiding false signals in choppy markets.
The strategy uses ATR-based trailing stops to let winners run and cut losses quickly.
Designed for ~25-35 trades/year per symbol to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    if n < 50:
        return np.zeros(n)
    
    # === HTF: 1d EMA for trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators ===
    # ATR for stoploss and position sizing
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete level)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 40  # Enough for Donchian(20) and EMA(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            stop_hit = False
            
            if position_side > 0:  # Long position
                # ATR trailing stop: exit if price drops 2.5*ATR from highest high
                stop_level = highest_since_entry - 2.5 * atr[i]
                if low[i] < stop_level:
                    stop_hit = True
                # Trend reversal exit: if price crosses below 1d EMA
                elif close[i] < ema_1d_aligned[i]:
                    stop_hit = True
            else:  # Short position
                # ATR trailing stop: exit if price rises 2.5*ATR from lowest low
                stop_level = lowest_since_entry + 2.5 * atr[i]
                if high[i] > stop_level:
                    stop_hit = True
                # Trend reversal exit: if price crosses above 1d EMA
                elif close[i] > ema_1d_aligned[i]:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                # Maintain position
                signals[i] = position_side * SIZE
            continue
        
        # --- Entry Logic (only when flat) ---
        # Volume spike: current volume > 1.8x 20-period average
        vol_spike = volume[i] > vol_ma[i] * 1.8 if vol_ma[i] > 0 else False
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_high[i]
        bearish_breakout = close[i] < donch_low[i]
        
        # Trend filter: price must be on correct side of 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Long entry: bullish breakout + volume spike + above 1d EMA
        if bullish_breakout and vol_spike and above_ema:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short entry: bearish breakout + volume spike + below 1d EMA
        elif bearish_breakout and vol_spike and below_ema:
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
Experiment #001: 4h Donchian(20) breakout + 1d EMA trend + Volume spike

HYPOTHESIS: In both bull and bear markets, strong directional moves begin with price breaking
out of the recent 4h Donchian channel (20-bar high/low). By requiring alignment with the
1d EMA trend (primary trend filter) and a volume spike (confirmation of participation),
we capture high-probability breakouts while avoiding false signals in choppy markets.
The strategy uses ATR-based trailing stops to let winners run and cut losses quickly.
Designed for ~25-35 trades/year per symbol to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    if n < 50:
        return np.zeros(n)
    
    # === HTF: 1d EMA for trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators ===
    # ATR for stoploss and position sizing
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete level)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 40  # Enough for Donchian(20) and EMA(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            stop_hit = False
            
            if position_side > 0:  # Long position
                # ATR trailing stop: exit if price drops 2.5*ATR from highest high
                stop_level = highest_since_entry - 2.5 * atr[i]
                if low[i] < stop_level:
                    stop_hit = True
                # Trend reversal exit: if price crosses below 1d EMA
                elif close[i] < ema_1d_aligned[i]:
                    stop_hit = True
            else:  # Short position
                # ATR trailing stop: exit if price rises 2.5*ATR from lowest low
                stop_level = lowest_since_entry + 2.5 * atr[i]
                if high[i] > stop_level:
                    stop_hit = True
                # Trend reversal exit: if price crosses above 1d EMA
                elif close[i] > ema_1d_aligned[i]:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                # Maintain position
                signals[i] = position_side * SIZE
            continue
        
        # --- Entry Logic (only when flat) ---
        # Volume spike: current volume > 1.8x 20-period average
        vol_spike = volume[i] > vol_ma[i] * 1.8 if vol_ma[i] > 0 else False
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_high[i]
        bearish_breakout = close[i] < donch_low[i]
        
        # Trend filter: price must be on correct side of 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Long entry: bullish breakout + volume spike + above 1d EMA
        if bullish_breakout and vol_spike and above_ema:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short entry: bearish breakout + volume spike + below 1d EMA
        elif bearish_breakout and vol_spike and below_ema:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals

</think>