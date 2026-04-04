#!/usr/bin/env python3
"""
Experiment #5474: 1h Donchian(20) breakout + 4h EMA(50) trend + volume confirmation
HYPOTHESIS: On 1h timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 1.5x average and aligned with 4h EMA(50) bias (price above EMA = bullish, below = bearish) 
captures momentum moves while avoiding false breakouts. Using 4h for signal direction reduces 
whipsaws vs pure 1h breakouts. Session filter (08-20 UTC) targets liquidity periods. 
Discrete position sizing (0.20) and ATR-based stoploss (2.0x ATR) control risk. Target: 60-150 
total trades over 4 years = 15-37/year for 1h. Works in bull markets via breakouts above rising 
EMA alignment and in bear markets via short breakdowns below falling EMA alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5474_1h_donchian20_4h_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA(50) trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        # Calculate EMA(50) on 4h close
        ema_4h = pd.Series(df_4h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        # Align to LTF (1h) with shift(1) for completed bars only
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
        # Trend bias: price above EMA = bullish, below = bearish
        price_above_ema = close > ema_4h_aligned
        price_below_ema = close < ema_4h_aligned
    else:
        ema_4h_aligned = np.full(n, np.nan)
        price_above_ema = np.full(n, False)
        price_below_ema = np.full(n, False)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 1h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14, 50)  # Donchian, volume avg, ATR warmup, EMA lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Trade during 08-20 UTC (major sessions) ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price moves below 4h EMA (trend weakening)
                if price <= stop_price or price <= donchian_low[i] or price < ema_4h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price moves above 4h EMA (trend weakening)
                if price >= stop_price or price >= donchian_high[i] or price > ema_4h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Entry conditions
        if breakout_up and volume_confirmed and price_above_ema[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and price_below_ema[i]:
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
Experiment #5474: 1h Donchian(20) breakout + 4h EMA(50) trend + volume confirmation
HYPOTHESIS: On 1h timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 1.5x average and aligned with 4h EMA(50) bias (price above EMA = bullish, below = bearish) 
captures momentum moves while avoiding false breakouts. Using 4h for signal direction reduces 
whipsaws vs pure 1h breakouts. Session filter (08-20 UTC) targets liquidity periods. 
Discrete position sizing (0.20) and ATR-based stoploss (2.0x ATR) control risk. Target: 60-150 
total trades over 4 years = 15-37/year for 1h. Works in bull markets via breakouts above rising 
EMA alignment and in bear markets via short breakdowns below falling EMA alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5474_1h_donchian20_4h_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA(50) trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        # Calculate EMA(50) on 4h close
        ema_4h = pd.Series(df_4h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        # Align to LTF (1h) with shift(1) for completed bars only
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
        # Trend bias: price above EMA = bullish, below = bearish
        price_above_ema = close > ema_4h_aligned
        price_below_ema = close < ema_4h_aligned
    else:
        ema_4h_aligned = np.full(n, np.nan)
        price_above_ema = np.full(n, False)
        price_below_ema = np.full(n, False)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 1h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14, 50)  # Donchian, volume avg, ATR warmup, EMA lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Trade during 08-20 UTC (major sessions) ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price moves below 4h EMA (trend weakening)
                if price <= stop_price or price <= donchian_low[i] or price < ema_4h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price moves above 4h EMA (trend weakening)
                if price >= stop_price or price >= donchian_high[i] or price > ema_4h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Entry conditions
        if breakout_up and volume_confirmed and price_above_ema[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and price_below_ema[i]:
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