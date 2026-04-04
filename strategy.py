#!/usr/bin/env python3
"""
Experiment #4831: 6h Camarilla Pivot + 1d Trend + Volume Spike
HYPOTHESIS: On 6h timeframe, Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 1d timeframe provide institutional support/resistance. Long when price breaks above R4 with 1d uptrend and volume spike (>2x). Short when price breaks below S4 with 1d downtrend and volume spike. Uses ATR(14) stoploss (2.5x trailing). Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (R4 breakouts with trend) and bear markets (S4 breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4831_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3, R4, S4) ===
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate pivot point
        pivot_1d = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        
        # Camarilla levels
        r3_1d = pivot_1d + range_1d * 1.1 / 4.0
        s3_1d = pivot_1d - range_1d * 1.1 / 4.0
        r4_1d = pivot_1d + range_1d * 1.1 / 2.0
        s4_1d = pivot_1d - range_1d * 1.1 / 2.0
    else:
        pivot_1d = np.array([])
        r3_1d = np.array([])
        s3_1d = np.array([])
        r4_1d = np.array([])
        s4_1d = np.array([])
    
    # Align HTF Camarilla levels to 6h timeframe
    if len(r3_1d) > 0:
        r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
        s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
        r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
        s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    else:
        r3_1d_aligned = np.full(n, np.nan)
        s3_1d_aligned = np.full(n, np.nan)
        r4_1d_aligned = np.full(n, np.nan)
        s4_1d_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA21 for trend filter ===
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF EMA21 to 6h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 21, 14)  # Volume MA, EMA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Camarilla breakout conditions with trend alignment
        breakout_long = (price >= r4_1d_aligned[i]) and (price > ema_1d_aligned[i]) and vol_confirm
        breakout_short = (price <= s4_1d_aligned[i]) and (price < ema_1d_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
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
Experiment #4831: 6h Camarilla Pivot + 1d Trend + Volume Spike
HYPOTHESIS: On 6h timeframe, Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 1d timeframe provide institutional support/resistance. Long when price breaks above R4 with 1d uptrend and volume spike (>2x). Short when price breaks below S4 with 1d downtrend and volume spike. Uses ATR(14) stoploss (2.5x trailing). Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (R4 breakouts with trend) and bear markets (S4 breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4831_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3, R4, S4) ===
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate pivot point
        pivot_1d = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        
        # Camarilla levels
        r3_1d = pivot_1d + range_1d * 1.1 / 4.0
        s3_1d = pivot_1d - range_1d * 1.1 / 4.0
        r4_1d = pivot_1d + range_1d * 1.1 / 2.0
        s4_1d = pivot_1d - range_1d * 1.1 / 2.0
    else:
        pivot_1d = np.array([])
        r3_1d = np.array([])
        s3_1d = np.array([])
        r4_1d = np.array([])
        s4_1d = np.array([])
    
    # Align HTF Camarilla levels to 6h timeframe
    if len(r3_1d) > 0:
        r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
        s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
        r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
        s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    else:
        r3_1d_aligned = np.full(n, np.nan)
        s3_1d_aligned = np.full(n, np.nan)
        r4_1d_aligned = np.full(n, np.nan)
        s4_1d_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA21 for trend filter ===
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF EMA21 to 6h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 21, 14)  # Volume MA, EMA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Camarilla breakout conditions with trend alignment
        breakout_long = (price >= r4_1d_aligned[i]) and (price > ema_1d_aligned[i]) and vol_confirm
        breakout_short = (price <= s4_1d_aligned[i]) and (price < ema_1d_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
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