#!/usr/bin/env python3
"""
Experiment #139: 6h Camarilla Pivot + 12h Volume Spike + Regime Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
combined with 12h volume confirmation and ADX regime filter captures institutional
activity around key pivot levels. In ranging markets (ADX<25), fade extremes at R3/S3.
In trending markets (ADX>25), breakout continuation at R4/S4. Discrete sizing (0.25)
and ATR trailing stop (2.0x) manage risk. Targets 12-25 trades/year on 6h timeframe.
Works in bull/bear by adapting to regime: mean revert in range, follow trend when strong.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_camarilla(high, low, close):
    """Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    
    r4 = close + range_ * 1.1 / 2.0
    r3 = close + range_ * 1.1 / 4.0
    s3 = close - range_ * 1.1 / 4.0
    s4 = close - range_ * 1.1 / 2.0
    
    return r3, r4, s3, s4

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume MA (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values.astype(np.float64)
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === 6h Indicators ===
    adx = calculate_adx(high, low, close, 14)
    atr_14 = pd.Series(np.maximum(high - low, 
                                  np.maximum(np.abs(high - np.roll(close, 1)),
                                             np.abs(low - np.roll(close, 1))))).ewm(
        span=14, min_periods=14, adjust=False).mean().values
    atr_14[0] = high[0] - low[0]  # First value
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Camarilla Pivot Levels (using previous bar) ---
        r3, r4, s3, s4 = calculate_camarilla(high[i-1], low[i-1], close[i-1])
        
        # --- 12h Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_12h_aligned[i] * 1.5 if vol_ma_12h_aligned[i] > 1e-10 else False
        
        # --- Regime Filter ---
        ranging = adx[i] < 25   # ADX < 25 = ranging market
        trending = adx[i] > 25  # ADX > 25 = trending market
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions based on regime
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price reaches opposite Camarilla level OR ADX drops
                    if close[i] <= s3 or adx[i] < 20:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price reaches opposite Camarilla level OR ADX drops
                    if close[i] >= r3 or adx[i] < 20:
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
        if ranging:
            # In ranging market: mean reversion at S3
            if low[i] <= s3 and close[i] > s3 and vol_ok:
                in_position = True
                position_side = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
        else:  # trending
            # In trending market: breakout continuation at R4
            if high[i] >= r4 and close[i] < r4 and vol_ok:
                in_position = True
                position_side = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
        
        # Short conditions:
        if ranging:
            # In ranging market: mean reversion at R3
            if high[i] >= r3 and close[i] < r3 and vol_ok:
                in_position = True
                position_side = -1
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
        else:  # trending
            # In trending market: breakdown continuation at S4
            if low[i] <= s4 and close[i] > s4 and vol_ok:
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
Experiment #139: 6h Camarilla Pivot + 12h Volume Spike + Regime Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
combined with 12h volume confirmation and ADX regime filter captures institutional
activity around key pivot levels. In ranging markets (ADX<25), fade extremes at R3/S3.
In trending markets (ADX>25), breakout continuation at R4/S4. Discrete sizing (0.25)
and ATR trailing stop (2.0x) manage risk. Targets 12-25 trades/year on 6h timeframe.
Works in bull/bear by adapting to regime: mean revert in range, follow trend when strong.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
   dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_camarilla(high, low, close):
    """Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    
    r4 = close + range_ * 1.1 / 2.0
    r3 = close + range_ * 1.1 / 4.0
    s3 = close - range_ * 1.1 / 4.0
    s4 = close - range_ * 1.1 / 2.0
    
    return r3, r4, s3, s4

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume MA (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values.astype(np.float64)
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === 6h Indicators ===
    adx = calculate_adx(high, low, close, 14)
    atr_14 = pd.Series(np.maximum(high - low, 
                                  np.maximum(np.abs(high - np.roll(close, 1)),
                                             np.abs(low - np.roll(close, 1))))).ewm(
        span=14, min_periods=14, adjust=False).mean().values
    atr_14[0] = high[0] - low[0]  # First value
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Camarilla Pivot Levels (using previous bar) ---
        r3, r4, s3, s4 = calculate_camarilla(high[i-1], low[i-1], close[i-1])
        
        # --- 12h Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_12h_aligned[i] * 1.5 if vol_ma_12h_aligned[i] > 1e-10 else False
        
        # --- Regime Filter ---
        ranging = adx[i] < 25   # ADX < 25 = ranging market
        trending = adx[i] > 25  # ADX > 25 = trending market
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions based on regime
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price reaches opposite Camarilla level OR ADX drops
                    if close[i] <= s3 or adx[i] < 20:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price reaches opposite Camarilla level OR ADX drops
                    if close[i] >= r3 or adx[i] < 20:
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
        if ranging:
            # In ranging market: mean reversion at S3
            if low[i] <= s3 and close[i] > s3 and vol_ok:
                in_position = True
                position_side = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
        else:  # trending
            # In trending market: breakout continuation at R4
            if high[i] >= r4 and close[i] < r4 and vol_ok:
                in_position = True
                position_side = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
        
        # Short conditions:
        if ranging:
            # In ranging market: mean reversion at R3
            if high[i] >= r3 and close[i] < r3 and vol_ok:
                in_position = True
                position_side = -1
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
        else:  # trending
            # In trending market: breakdown continuation at S4
            if low[i] <= s4 and close[i] > s4 and vol_ok:
                in_position = True
                position_side = -1
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals