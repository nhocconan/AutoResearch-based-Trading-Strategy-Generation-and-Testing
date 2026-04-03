#!/usr/bin/env python3
"""
Experiment #019: 6h Williams %R + 1d Elder Ray + Volume Confirmation
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, while 1d Elder Ray (Bull/Bear Power) confirms the underlying trend direction. Volume spikes (>1.5x average) validate institutional participation. This combination works in both bull and bear markets by fading extremes with trend alignment. Target: 75-150 total trades over 4 years (19-37/year). Discrete sizing (0.25) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_019_6h_williamsr_1d_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Elder Ray (Bull/Bear Power) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    if len(df_1d) >= 13:
        ema_13 = df_1d['close'].ewm(span=13, min_periods=13, adjust=False).mean().values
        bull_power = (df_1d['high'].values - ema_13)
        bear_power = (df_1d['low'].values - ema_13)
        # Trend: Bull Power > 0 = bullish, Bear Power < 0 = bearish
        elder_bull_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
        elder_bear_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    else:
        elder_bull_aligned = np.full(n, np.nan)
        elder_bear_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    denominator = highest_high_14 - lowest_low_14
    mask = denominator != 0
    williams_r[mask] = ((highest_high_14[mask] - close[mask]) / denominator[mask]) * -100
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.full(n, 1.0)
    mask_vol = ~np.isnan(vol_ma) & (vol_ma > 0)
    vol_ratio[mask_vol] = volume[mask_vol] / vol_ma[mask_vol]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 14-period Williams %R + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or
            np.isnan(williams_r[i]) or np.isnan(elder_bull_aligned[i]) or
            np.isnan(elder_bear_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        bull_power = elder_bull_aligned[i]
        bear_power = elder_bear_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Williams %R Conditions: Oversold < -80, Overbought > -20 ---
        oversold = wr < -80
        overbought = wr > -20
        
        # --- Elder Ray Trend Conditions ---
        bullish_trend = bull_power > 0
        bearish_trend = bear_power < 0
        
        # --- Exit Logic: ATR-based stoploss (using 2.0*ATR for tighter stops) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32 hours on 6h)
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Williams %R oversold AND 1d Bull Power positive (bullish trend)
            if oversold and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Williams %R overbought AND 1d Bear Power negative (bearish trend)
            elif overbought and bearish_trend:
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
Experiment #019: 6h Williams %R + 1d Elder Ray + Volume Confirmation
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, while 1d Elder Ray (Bull/Bear Power) confirms the underlying trend direction. Volume spikes (>1.5x average) validate institutional participation. This combination works in both bull and bear markets by fading extremes with trend alignment. Target: 75-150 total trades over 4 years (19-37/year). Discrete sizing (0.25) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_019_6h_williamsr_1d_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Elder Ray (Bull/Bear Power) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    if len(df_1d) >= 13:
        ema_13 = df_1d['close'].ewm(span=13, min_periods=13, adjust=False).mean().values
        bull_power = (df_1d['high'].values - ema_13)
        bear_power = (df_1d['low'].values - ema_13)
        # Trend: Bull Power > 0 = bullish, Bear Power < 0 = bearish
        elder_bull_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
        elder_bear_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    else:
        elder_bull_aligned = np.full(n, np.nan)
        elder_bear_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    denominator = highest_high_14 - lowest_low_14
    mask = denominator != 0
    williams_r[mask] = ((highest_high_14[mask] - close[mask]) / denominator[mask]) * -100
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.full(n, 1.0)
    mask_vol = ~np.isnan(vol_ma) & (vol_ma > 0)
    vol_ratio[mask_vol] = volume[mask_vol] / vol_ma[mask_vol]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 14-period Williams %R + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or
            np.isnan(williams_r[i]) or np.isnan(elder_bull_aligned[i]) or
            np.isnan(elder_bear_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        bull_power = elder_bull_aligned[i]
        bear_power = elder_bear_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Williams %R Conditions: Oversold < -80, Overbought > -20 ---
        oversold = wr < -80
        overbought = wr > -20
        
        # --- Elder Ray Trend Conditions ---
        bullish_trend = bull_power > 0
        bearish_trend = bear_power < 0
        
        # --- Exit Logic: ATR-based stoploss (using 2.0*ATR for tighter stops) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32 hours on 6h)
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Williams %R oversold AND 1d Bull Power positive (bullish trend)
            if oversold and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Williams %R overbought AND 1d Bear Power negative (bearish trend)
            elif overbought and bearish_trend:
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