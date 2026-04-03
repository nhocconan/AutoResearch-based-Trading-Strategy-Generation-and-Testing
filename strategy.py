#!/usr/bin/env python3
"""
Experiment #211: 6h Ichimoku Cloud + 1d Weekly Pivot + Volume Confirmation

HYPOTHESIS: Ichimoku cloud (Tenkan/Kijun cross + price relative to cloud) from 6h combined with 
1d weekly pivot levels (S1/S2/R1/R2) acts as a strong confluence filter for breakout/continuation trades.
Volume confirmation (>1.5x 20-period average volume on 1d) ensures momentum validity.
Ichimoku works in both bull/bear markets via cloud support/resistance and TK crosses capturing momentum shifts.
Weekly pivots provide institutional reference points. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_1d_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and volume filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot levels from 1d (using prior week's high/low/close)
    # We calculate weekly pivot on daily data, assuming 5 trading days per week
    week_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close).rolling(window=5, min_periods=5).last().values  # last close of week
    week_pivot = (week_high + week_low + week_close) / 3.0
    week_r1 = 2 * week_pivot - week_low
    week_s1 = 2 * week_pivot - week_high
    week_r2 = week_pivot + (week_high - week_low)
    week_s2 = week_pivot - (week_high - week_low)
    
    # Align weekly pivot levels to 6h
    week_r1_6h = align_htf_to_ltf(prices, df_1d, week_r1)
    week_s1_6h = align_htf_to_ltf(prices, df_1d, week_s1)
    week_r2_6h = align_htf_to_ltf(prices, df_1d, week_r2)
    week_s2_6h = align_htf_to_ltf(prices, df_1d, week_s2)
    
    # Volume spike filter on 1d (>1.5x 20-day average volume)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * avg_vol_1d)
    vol_spike_1d_6h = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_a = ((tenkan + kijun) / 2.0)
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Current Ichimoku cloud boundaries (using previously calculated Senkou spans)
    # Since Senkou spans are plotted 26 periods ahead, the current cloud uses values from 26 periods ago
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = senkou_a[0]  # fill NaN with first value
    senkou_b_lag[:26] = senkou_b[0]
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_lag, senkou_b_lag)
    cloud_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
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
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(vol_spike_1d_6h[i]) or
            np.isnan(week_r1_6h[i]) or np.isnan(week_s1_6h[i])):
            signals[i] = 0.0
            continue
        
        # --- Ichimoku Signals ---
        # Price above cloud = bullish bias
        price_above_cloud = close[i] > cloud_top[i]
        # Price below cloud = bearish bias
        price_below_cloud = close[i] < cloud_bottom[i]
        # TK cross bullish: Tenkan crosses above Kijun
        tk_bullish = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1]) if i > 0 else False
        # TK cross bearish: Tenkan crosses below Kijun
        tk_bearish = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1]) if i > 0 else False
        
        # --- Weekly Pivot Levels ---
        # Near weekly support/resistance (within 0.5% of level)
        near_week_r1 = abs(close[i] - week_r1_6h[i]) / close[i] < 0.005
        near_week_s1 = abs(close[i] - week_s1_6h[i]) / close[i] < 0.005
        break_above_r2 = close[i] > week_r2_6h[i]
        break_below_s2 = close[i] < week_s2_6h[i]
        
        # --- Position Management ---
        if in_position:
            # Stoploss: 2.5 * ATR against position (simplified using price/cloud distance)
            if position_side > 0:  # Long
                # Exit if price falls below cloud or breaks below weekly S1
                if close[i] < cloud_bottom[i] or close[i] < week_s1_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                # Exit if price rises above cloud or breaks above weekly R1
                if close[i] > cloud_top[i] or close[i] > week_r1_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Still in position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price above cloud + TK bullish cross + near weekly support OR break above R2 + volume
        if (price_above_cloud and tk_bullish and (near_week_s1 or break_above_r2)) and vol_spike_1d_6h[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Price below cloud + TK bearish cross + near weekly resistance OR break below S2 + volume
        elif (price_below_cloud and tk_bearish and (near_week_r1 or break_below_s2)) and vol_spike_1d_6h[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #211: 6h Ichimoku Cloud + 1d Weekly Pivot + Volume Confirmation

HYPOTHESIS: Ichimoku cloud (Tenkan/Kijun cross + price relative to cloud) from 6h combined with 
1d weekly pivot levels (S1/S2/R1/R2) acts as a strong confluence filter for breakout/continuation trades.
Volume confirmation (>1.5x 20-period average volume on 1d) ensures momentum validity.
Ichimoku works in both bull/bear markets via cloud support/resistance and TK crosses capturing momentum shifts.
Weekly pivots provide institutional reference points. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_1d_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and volume filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot levels from 1d (using prior week's high/low/close)
    # We calculate weekly pivot on daily data, assuming 5 trading days per week
    week_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close).rolling(window=5, min_periods=5).last().values  # last close of week
    week_pivot = (week_high + week_low + week_close) / 3.0
    week_r1 = 2 * week_pivot - week_low
    week_s1 = 2 * week_pivot - week_high
    week_r2 = week_pivot + (week_high - week_low)
    week_s2 = week_pivot - (week_high - week_low)
    
    # Align weekly pivot levels to 6h
    week_r1_6h = align_htf_to_ltf(prices, df_1d, week_r1)
    week_s1_6h = align_htf_to_ltf(prices, df_1d, week_s1)
    week_r2_6h = align_htf_to_ltf(prices, df_1d, week_r2)
    week_s2_6h = align_htf_to_ltf(prices, df_1d, week_s2)
    
    # Volume spike filter on 1d (>1.5x 20-day average volume)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * avg_vol_1d)
    vol_spike_1d_6h = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_a = ((tenkan + kijun) / 2.0)
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Current Ichimoku cloud boundaries (using previously calculated Senkou spans)
    # Since Senkou spans are plotted 26 periods ahead, the current cloud uses values from 26 periods ago
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = senkou_a[0]  # fill NaN with first value
    senkou_b_lag[:26] = senkou_b[0]
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_lag, senkou_b_lag)
    cloud_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
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
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(vol_spike_1d_6h[i]) or
            np.isnan(week_r1_6h[i]) or np.isnan(week_s1_6h[i])):
            signals[i] = 0.0
            continue
        
        # --- Ichimoku Signals ---
        # Price above cloud = bullish bias
        price_above_cloud = close[i] > cloud_top[i]
        # Price below cloud = bearish bias
        price_below_cloud = close[i] < cloud_bottom[i]
        # TK cross bullish: Tenkan crosses above Kijun
        tk_bullish = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1]) if i > 0 else False
        # TK cross bearish: Tenkan crosses below Kijun
        tk_bearish = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1]) if i > 0 else False
        
        # --- Weekly Pivot Levels ---
        # Near weekly support/resistance (within 0.5% of level)
        near_week_r1 = abs(close[i] - week_r1_6h[i]) / close[i] < 0.005
        near_week_s1 = abs(close[i] - week_s1_6h[i]) / close[i] < 0.005
        break_above_r2 = close[i] > week_r2_6h[i]
        break_below_s2 = close[i] < week_s2_6h[i]
        
        # --- Position Management ---
        if in_position:
            # Stoploss: 2.5 * ATR against position (simplified using price/cloud distance)
            if position_side > 0:  # Long
                # Exit if price falls below cloud or breaks below weekly S1
                if close[i] < cloud_bottom[i] or close[i] < week_s1_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                # Exit if price rises above cloud or breaks above weekly R1
                if close[i] > cloud_top[i] or close[i] > week_r1_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Still in position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price above cloud + TK bullish cross + near weekly support OR break above R2 + volume
        if (price_above_cloud and tk_bullish and (near_week_s1 or break_above_r2)) and vol_spike_1d_6h[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Price below cloud + TK bearish cross + near weekly resistance OR break below S2 + volume
        elif (price_below_cloud and tk_bearish and (near_week_r1 or break_below_s2)) and vol_spike_1d_6h[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals