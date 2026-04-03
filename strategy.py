#!/usr/bin/env python3
"""
Experiment #031: 6h Ichimoku Cloud + 1d TK Cross + Volume Spike

HYPOTHESIS: Ichimoku system on 6h timeframe provides robust trend identification 
via cloud (Senkou Span A/B) and momentum via TK Cross (Tenkan/Kijun). 
Using 1d timeframe for TK Cross direction filter ensures alignment with higher timeframe trend. 
Volume confirmation (1.5x average) filters breakouts. 
Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag in choppy/bear markets.
Ichimoku works in both bull/bear: cloud acts as dynamic support/resistance, TK cross captures momentum shifts.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_1d_tkcross_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper min_periods."""
    return pd.Series(values).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span."""
    n = len(high)
    if n < 52:
        return (np.full(n, np.nan), np.full(n, np.nan), 
                np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = np.roll(close, -26)  # Will handle alignment in main logic
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d TK Cross for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values.astype(np.float64)
    low_1d = df_1d['low'].values.astype(np.float64)
    close_1d = df_1d['close'].values.astype(np.float64)
    
    # Calculate 1d Ichimoku for TK Cross
    tenkan_1d, kijun_1d, _, _, _ = calculate_ichimoku(high_1d, low_1d, close_1d)
    tk_cross_1d = tenkan_1d - kijun_1d  # Positive = bullish cross, Negative = bearish
    tk_cross_1d_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_1d)
    
    # === 6h Ichimoku ===
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high, low, close)
    
    # 6h ATR for stoploss
    atr_6h = calculate_atr(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Cloud calculation (Senkou Span A/B) - note: these are already shifted 26 periods ahead in calculation
    # For current cloud, we need the values that were calculated 26 periods ago
    senkou_a_current = np.roll(senkou_a_6h, 26)  # Shift back to align with current price
    senkou_b_current = np.roll(senkou_b_6h, 26)
    # Handle first 26 values (will be NaN due to roll)
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for Ichimoku calculations (52 + 26 shift)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_6h[i]) or np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tk_cross_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d TK Cross Trend Filter ---
        tk_bullish = tk_cross_1d_aligned[i] > 0  # Tenkan > Kijun on 1d
        tk_bearish = tk_cross_1d_aligned[i] < 0  # Tenkan < Kijun on 1d
        
        # --- Price vs Cloud ---
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        price_in_cloud = ~(price_above_cloud | price_below_cloud)
        
        # --- TK Cross on 6h (momentum) ---
        tk_cross_6h_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_cross_6h_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_6h[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_6h[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: 
            # 1. Price re-enters cloud (trend weakening)
            # 2. TK Cross reverses on 6h
            # 3. 1d TK Cross reverses (higher timeframe trend change)
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price re-enters cloud OR 6h TK cross bearish OR 1d TK cross bearish
                    if price_in_cloud or not tk_cross_6h_bullish or not tk_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price re-enters cloud OR 6h TK cross bullish OR 1d TK cross bullish
                    if price_in_cloud or not tk_cross_6h_bearish or not tk_bearish:
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
        # Price above cloud (bullish trend) + 6h TK cross bullish + 1d TK cross bullish + volume
        if price_above_cloud and tk_cross_6h_bullish and tk_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Price below cloud (bearish trend) + 6h TK cross bearish + 1d TK cross bearish + volume
        elif price_below_cloud and tk_cross_6h_bearish and tk_bearish and vol_ok:
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
Experiment #031: 6h Ichimoku Cloud + 1d TK Cross + Volume Spike

HYPOTHESIS: Ichimoku system on 6h timeframe provides robust trend identification 
via cloud (Senkou Span A/B) and momentum via TK Cross (Tenkan/Kijun). 
Using 1d timeframe for TK Cross direction filter ensures alignment with higher timeframe trend. 
Volume confirmation (1.5x average) filters breakouts. 
Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag in choppy/bear markets.
Ichimoku works in both bull/bear: cloud acts as dynamic support/resistance, TK cross captures momentum shifts.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_1d_tkcross_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper min_periods."""
    return pd.Series(values).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span."""
    n = len(high)
    if n < 52:
        return (np.full(n, np.nan), np.full(n, np.nan), 
                np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = np.roll(close, -26)  # Will handle alignment in main logic
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d TK Cross for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values.astype(np.float64)
    low_1d = df_1d['low'].values.astype(np.float64)
    close_1d = df_1d['close'].values.astype(np.float64)
    
    # Calculate 1d Ichimoku for TK Cross
    tenkan_1d, kijun_1d, _, _, _ = calculate_ichimoku(high_1d, low_1d, close_1d)
    tk_cross_1d = tenkan_1d - kijun_1d  # Positive = bullish cross, Negative = bearish
    tk_cross_1d_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_1d)
    
    # === 6h Ichimoku ===
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high, low, close)
    
    # 6h ATR for stoploss
    atr_6h = calculate_atr(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Cloud calculation (Senkou Span A/B) - note: these are already shifted 26 periods ahead in calculation
    # For current cloud, we need the values that were calculated 26 periods ago
    senkou_a_current = np.roll(senkou_a_6h, 26)  # Shift back to align with current price
    senkou_b_current = np.roll(senkou_b_6h, 26)
    # Handle first 26 values (will be NaN due to roll)
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for Ichimoku calculations (52 + 26 shift)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_6h[i]) or np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tk_cross_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d TK Cross Trend Filter ---
        tk_bullish = tk_cross_1d_aligned[i] > 0  # Tenkan > Kijun on 1d
        tk_bearish = tk_cross_1d_aligned[i] < 0  # Tenkan < Kijun on 1d
        
        # --- Price vs Cloud ---
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        price_in_cloud = ~(price_above_cloud | price_below_cloud)
        
        # --- TK Cross on 6h (momentum) ---
        tk_cross_6h_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_cross_6h_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_6h[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_6h[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: 
            # 1. Price re-enters cloud (trend weakening)
            # 2. TK Cross reverses on 6h
            # 3. 1d TK Cross reverses (higher timeframe trend change)
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price re-enters cloud OR 6h TK cross bearish OR 1d TK cross bearish
                    if price_in_cloud or not tk_cross_6h_bullish or not tk_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price re-enters cloud OR 6h TK cross bullish OR 1d TK cross bullish
                    if price_in_cloud or not tk_cross_6h_bearish or not tk_bearish:
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
        # Price above cloud (bullish trend) + 6h TK cross bullish + 1d TK cross bullish + volume
        if price_above_cloud and tk_cross_6h_bullish and tk_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Price below cloud (bearish trend) + 6h TK cross bearish + 1d TK cross bearish + volume
        elif price_below_cloud and tk_cross_6h_bearish and tk_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>