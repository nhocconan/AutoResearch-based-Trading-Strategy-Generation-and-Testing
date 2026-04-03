#!/usr/bin/env python3
"""
Experiment #138: 1d Donchian Breakout + 1w Volume + ATR Stoploss

HYPOTHESIS: Daily Donchian(20) breakouts capture medium-term trends, confirmed by weekly volume surge.
Uses 1d primary timeframe with 1h HTF for volume confirmation to avoid false breakouts.
ATR-based stoploss limits drawdown. Designed to work in both bull (breakouts up) and bear (breakdowns down).
Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume confirmation ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly volume average (20-period)
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === 1d Indicators ===
    # Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # ATR for stoploss and position sizing
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        vol_ratio = volume[i] / vol_ma_1w_aligned[i] if vol_ma_1w_aligned[i] > 0 else 0
        volume_confirmed = vol_ratio > 1.5  # 50% above average weekly volume
        
        # --- Position Management ---
        if in_position:
            # Stoploss: 2 * ATR against position
            if position_side > 0:  # Long
                if close[i] < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if close[i] > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Maintain position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: price closes above Donchian upper with volume confirmation
        if close[i] > donch_upper[i] and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short breakdown: price closes below Donchian lower with volume confirmation
        elif close[i] < donch_lower[i] and volume_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #138: 1d Donchian Breakout + 1w Volume + ATR Stoploss

HYPOTHESIS: Daily Donchian(20) breakouts capture medium-term trends, confirmed by weekly volume surge.
Uses 1d primary timeframe with 1h HTF for volume confirmation to avoid false breakouts.
ATR-based stoploss limits drawdown. Designed to work in both bull (breakouts up) and bear (breakdowns down).
Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume confirmation ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly volume average (20-period)
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === 1d Indicators ===
    # Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # ATR for stoploss and position sizing
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        vol_ratio = volume[i] / vol_ma_1w_aligned[i] if vol_ma_1w_aligned[i] > 0 else 0
        volume_confirmed = vol_ratio > 1.5  # 50% above average weekly volume
        
        # --- Position Management ---
        if in_position:
            # Stoploss: 2 * ATR against position
            if position_side > 0:  # Long
                if close[i] < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if close[i] > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Maintain position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: price closes above Donchian upper with volume confirmation
        if close[i] > donch_upper[i] and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short breakdown: price closes below Donchian lower with volume confirmation
        elif close[i] < donch_lower[i] and volume_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals

</think>