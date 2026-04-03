#!/usr/bin/env python3
"""
Experiment #059: 6h Williams %R(14) + 12h Elder Ray (Bull/Bear Power) + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, while 12h Elder Ray confirms trend strength via bull/bear power. Volume spike (>2.0x average) filters false signals. This mean-reversion-with-trend-filter approach works in both bull and bear markets by fading extremes only when aligned with higher timeframe momentum. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_059_6h_williamsr14_12h_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Elder Ray (Bull/Bear Power) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Elder Ray on 12h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_12h = pd.Series(df_12h['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_12h = df_12h['high'].values - ema13_12h
    bear_power_12h = df_12h['low'].values - ema13_12h
    
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # === 6h Indicators: Williams %R(14) ===
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Williams %R stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Conditions: Oversold < -80, Overbought > -20 ---
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        
        # --- Elder Ray Trend Confirmation: Bull Power > 0 = uptrend, Bear Power < 0 = downtrend ---
        bull_power_pos = bull_power_12h_aligned[i] > 0
        bear_power_neg = bear_power_12h_aligned[i] < 0
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral range (-50 to -50) with volume
                if williams_r[i] > -50 and williams_r[i] < -50 and volume_spike:  # This condition is never true, fixing
                    # Actually exit when WR crosses -50 (mean reversion complete)
                    if williams_r[i] > -50:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral range with volume
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold AND 12h Bull Power positive (uptrend) AND volume spike
        if wr_oversold and bull_power_pos and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Williams %R overbought AND 12h Bear Power negative (downtrend) AND volume spike
        elif wr_overbought and bear_power_neg and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #059: 6h Williams %R(14) + 12h Elder Ray (Bull/Bear Power) + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, while 12h Elder Ray confirms trend strength via bull/bear power. Volume spike (>2.0x average) filters false signals. This mean-reversion-with-trend-filter approach works in both bull and bear markets by fading extremes only when aligned with higher timeframe momentum. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_059_6h_williamsr14_12h_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Elder Ray (Bull/Bear Power) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Elder Ray on 12h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_12h = pd.Series(df_12h['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_12h = df_12h['high'].values - ema13_12h
    bear_power_12h = df_12h['low'].values - ema13_12h
    
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # === 6h Indicators: Williams %R(14) ===
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Williams %R stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Conditions: Oversold < -80, Overbought > -20 ---
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        
        # --- Elder Ray Trend Confirmation: Bull Power > 0 = uptrend, Bear Power < 0 = downtrend ---
        bull_power_pos = bull_power_12h_aligned[i] > 0
        bear_power_neg = bear_power_12h_aligned[i] < 0
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral range (-50) with volume
                if williams_r[i] > -50 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit when Williams %R returns to neutral range with volume
                if williams_r[i] < -50 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold AND 12h Bull Power positive (uptrend) AND volume spike
        if wr_oversold and bull_power_pos and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Williams %R overbought AND 12h Bear Power negative (downtrend) AND volume spike
        elif wr_overbought and bear_power_neg and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals