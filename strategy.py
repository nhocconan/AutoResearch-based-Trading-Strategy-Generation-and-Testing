#!/usr/bin/env python3
"""
Experiment #5135: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly Camarilla pivot bias (R3/S3 levels) capture institutional momentum. Volume > 2.0x average confirms participation. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets (breakouts above R3 with weekly uptrend) and bear markets (breakdowns below S3 with weekly downtrend). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5135_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute HTF: 1d data for weekly Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Camarilla Pivot Points (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data (prior completed week)
        # We'll use rolling window of 5 days to approximate weekly OHLC
        # Note: This is an approximation; for exact weekly we'd need to resample properly
        # but we avoid resampling per rules by using daily data with 5-day window
        high_5d = pd.Series(high).rolling(window=5*24//6, min_periods=5).max().values  # 5 days in 6h bars
        low_5d = pd.Series(low).rolling(window=5*24//6, min_periods=5).min().values
        close_5d = pd.Series(close).rolling(window=5*24//6, min_periods=5).last().values
        
        # Weekly pivot calculation (using prior week's OHLC)
        # Shift by 1 week to avoid look-ahead
        weekly_high = np.roll(high_5d, 5*24//6)
        weekly_low = np.roll(low_5d, 5*24//6)
        weekly_close = np.roll(close_5d, 5*24//6)
        
        # Camarilla levels
        pivot = (weekly_high + weekly_low + weekly_close) / 3
        range_ = weekly_high - weekly_low
        r3 = pivot + range_ * 1.1/2
        s3 = pivot - range_ * 1.1/2
        r4 = pivot + range_ * 1.1
        s4 = pivot - range_ * 1.1
        
        # Align to 6h timeframe
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (2.0x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with weekly Camarilla pivot bias
        # Long: Donchian breakout above R3 + price > R3 (bullish bias)
        # Short: Donchian breakdown below S3 + price < S3 (bearish bias)
        breakout_long = (price >= high_roll[i]) and (price > r3_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < s3_aligned[i]) and vol_confirm
        
        # Additional filter: avoid trading in extreme overbought/oversold (R4/S4)
        # Only allow longs below R4, shorts above S4 to prevent chasing extremes
        breakout_long = breakout_long and (price < r4_aligned[i])
        breakout_short = breakout_short and (price > s4_aligned[i])
        
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
Experiment #5135: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly Camarilla pivot bias (R3/S3 levels) capture institutional momentum. Volume > 2.0x average confirms participation. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets (breakouts above R3 with weekly uptrend) and bear markets (breakdowns below S3 with weekly downtrend). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5135_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute HTF: 1d data for weekly Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Camarilla Pivot Points (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data (prior completed week)
        # We'll use rolling window of 5 days to approximate weekly OHLC
        # Note: This is an approximation; for exact weekly we'd need to resample properly
        # but we avoid resampling per rules by using daily data with 5-day window
        high_5d = pd.Series(high).rolling(window=5*24//6, min_periods=5).max().values  # 5 days in 6h bars
        low_5d = pd.Series(low).rolling(window=5*24//6, min_periods=5).min().values
        close_5d = pd.Series(close).rolling(window=5*24//6, min_periods=5).last().values
        
        # Weekly pivot calculation (using prior week's OHLC)
        # Shift by 1 week to avoid look-ahead
        weekly_high = np.roll(high_5d, 5*24//6)
        weekly_low = np.roll(low_5d, 5*24//6)
        weekly_close = np.roll(close_5d, 5*24//6)
        
        # Camarilla levels
        pivot = (weekly_high + weekly_low + weekly_close) / 3
        range_ = weekly_high - weekly_low
        r3 = pivot + range_ * 1.1/2
        s3 = pivot - range_ * 1.1/2
        r4 = pivot + range_ * 1.1
        s4 = pivot - range_ * 1.1
        
        # Align to 6h timeframe
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (2.0x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with weekly Camarilla pivot bias
        # Long: Donchian breakout above R3 + price > R3 (bullish bias)
        # Short: Donchian breakdown below S3 + price < S3 (bearish bias)
        breakout_long = (price >= high_roll[i]) and (price > r3_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < s3_aligned[i]) and vol_confirm
        
        # Additional filter: avoid trading in extreme overbought/oversold (R4/S4)
        # Only allow longs below R4, shorts above S4 to prevent chasing extremes
        breakout_long = breakout_long and (price < r4_aligned[i])
        breakout_short = breakout_short and (price > s4_aligned[i])
        
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