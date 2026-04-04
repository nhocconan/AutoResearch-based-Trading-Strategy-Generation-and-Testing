#!/usr/bin/env python3
"""
Experiment #5034: 1h Donchian(20) Breakout + 4h/1d EMA Alignment + Volume Spike
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts aligned with 4h EMA20 trend and 1d EMA50 filter capture momentum with controlled frequency. 
4h EMA20 provides intermediate trend direction, 1d EMA50 acts as regime filter (bull/bear). Volume > 1.5x average confirms participation. 
Position size 0.20 to manage drawdown. Designed for 15-37 trades/year on 1h timeframe to minimize fee drag while maintaining statistical significance.
Works in bull markets (breakouts above rising EMAs) and bear markets (breakdowns below falling EMAs with volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5034_1h_donchian20_4h_ema20_1d_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA20 for trend direction ===
    if len(df_4h) >= 20:
        close_4h = df_4h['close'].values
        ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA50 for regime filter ===
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14)  # Donchian, Vol MA, ATR, EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend alignment: 4h EMA20 slope (using previous value)
        ema_4h_now = ema_4h_aligned[i]
        ema_4h_prev = ema_4h_aligned[i-1] if i > 0 else ema_4h_now
        ema_4h_rising = ema_4h_now > ema_4h_prev
        ema_4h_falling = ema_4h_now < ema_4h_prev
        
        # Regime filter: 1d EMA50 position
        price_vs_1d_ema = price > ema_1d_aligned[i]
        
        # Donchian breakout conditions with trend and regime alignment
        # Long: Donchian breakout above + 4h EMA rising + price above 1d EMA50 (bull regime)
        # Short: Donchian breakdown below + 4h EMA falling + price below 1d EMA50 (bear regime)
        breakout_long = (price >= high_roll[i]) and ema_4h_rising and price_vs_1d_ema and vol_confirm
        breakout_short = (price <= low_roll[i]) and ema_4h_falling and (not price_vs_1d_ema) and vol_confirm
        
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
Experiment #5034: 1h Donchian(20) Breakout + 4h/1d EMA Alignment + Volume Spike
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts aligned with 4h EMA20 trend and 1d EMA50 filter capture momentum with controlled frequency. 
4h EMA20 provides intermediate trend direction, 1d EMA50 acts as regime filter (bull/bear). Volume > 1.5x average confirms participation. 
Position size 0.20 to manage drawdown. Designed for 15-37 trades/year on 1h timeframe to minimize fee drag while maintaining statistical significance.
Works in bull markets (breakouts above rising EMAs) and bear markets (breakdowns below falling EMAs with volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5034_1h_donchian20_4h_ema20_1d_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA20 for trend direction ===
    if len(df_4h) >= 20:
        close_4h = df_4h['close'].values
        ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA50 for regime filter ===
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14)  # Donchian, Vol MA, ATR, EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend alignment: 4h EMA20 slope (using previous value)
        ema_4h_now = ema_4h_aligned[i]
        ema_4h_prev = ema_4h_aligned[i-1] if i > 0 else ema_4h_now
        ema_4h_rising = ema_4h_now > ema_4h_prev
        ema_4h_falling = ema_4h_now < ema_4h_prev
        
        # Regime filter: 1d EMA50 position
        price_vs_1d_ema = price > ema_1d_aligned[i]
        
        # Donchian breakout conditions with trend and regime alignment
        # Long: Donchian breakout above + 4h EMA rising + price above 1d EMA50 (bull regime)
        # Short: Donchian breakdown below + 4h EMA falling + price below 1d EMA50 (bear regime)
        breakout_long = (price >= high_roll[i]) and ema_4h_rising and price_vs_1d_ema and vol_confirm
        breakout_short = (price <= low_roll[i]) and ema_4h_falling and (not price_vs_1d_ema) and vol_confirm
        
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