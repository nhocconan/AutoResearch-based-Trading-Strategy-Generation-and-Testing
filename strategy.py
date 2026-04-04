#!/usr/bin/env python3
"""
Experiment #5499: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 2.0x average and aligned with 12-hour pivot bias (based on previous day's Camarilla 
pivot levels) captures strong momentum moves while avoiding false breakouts. The 12h pivot 
provides intermediate-term structure from higher timeframe, reducing whipsaws in both bull and 
bear markets. Discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) control risk. 
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining 
statistical significance. Works in bull markets via breakouts above rising pivot bias and in 
bear markets via short breakdowns below falling pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5499_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for pivot bias (using previous day's Camarilla) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla pivot levels from previous 12h bar (H1, L1, C1)
        # We use the completed 12h bar's data to calculate pivot for current bar
        h_12h = df_12h['high'].values
        l_12h = df_12h['low'].values
        c_12h = df_12h['close'].values
        
        # Camarilla pivot: P = (H1 + L1 + C1) / 3
        pivot_12h = (h_12h + l_12h + c_12h) / 3.0
        # Range = H1 - L1
        range_12h = h_12h - l_12h
        # Resistance levels: R3 = C1 + (Range * 1.1/4), R4 = C1 + (Range * 1.1/2)
        r3_12h = c_12h + (range_12h * 1.1 / 4)
        r4_12h = c_12h + (range_12h * 1.1 / 2)
        # Support levels: S3 = C1 - (Range * 1.1/4), S4 = C1 - (Range * 1.1/2)
        s3_12h = c_12h - (range_12h * 1.1 / 4)
        s4_12h = c_12h - (range_12h * 1.1 / 2)
        
        # Pivot bias: price above R3 = bullish bias, below S3 = bearish bias
        # We use the aligned values to check if current price is above/below these levels
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
        r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
        r4_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
        s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
        s4_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
        
        # Price above R3 = bullish bias, below S3 = bearish bias
        # Price between S3 and R3 = neutral (no bias)
        price_above_r3 = close > r3_aligned
        price_below_s3 = close < s3_aligned
        # For breakout continuation, we also check if price is above R4 or below S4
        price_above_r4 = close > r4_aligned
        price_below_s4 = close < s4_aligned
    else:
        # Not enough data - neutral bias
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        price_above_r3 = np.full(n, False)
        price_below_s3 = np.full(n, False)
        price_above_r4 = np.full(n, False)
        price_below_s4 = np.full(n, False)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14, 2)  # Donchian, volume avg, ATR warmup, 12h pivot lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
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
                # 3. Price moves below R3 (pivot bias weakening)
                if price <= stop_price or price <= donchian_low[i] or price < r3_aligned[i]:
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
                # 3. Price moves above S3 (pivot bias weakening)
                if price >= stop_price or price >= donchian_high[i] or price > s3_aligned[i]:
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
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Entry conditions with pivot bias
        # Long: breakout up + volume confirmed + price above R3 (bullish bias)
        # Short: breakout down + volume confirmed + price below S3 (bearish bias)
        if breakout_up and volume_confirmed and price_above_r3[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and price_below_s3[i]:
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
Experiment #5499: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel with 
volume > 2.0x average and aligned with 12-hour pivot bias (based on previous day's Camarilla 
pivot levels) captures strong momentum moves while avoiding false breakouts. The 12h pivot 
provides intermediate-term structure from higher timeframe, reducing whipsaws in both bull and 
bear markets. Discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) control risk. 
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining 
statistical significance. Works in bull markets via breakouts above rising pivot bias and in 
bear markets via short breakdowns below falling pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5499_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for pivot bias (using previous day's Camarilla) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla pivot levels from previous 12h bar (H1, L1, C1)
        # We use the completed 12h bar's data to calculate pivot for current bar
        h_12h = df_12h['high'].values
        l_12h = df_12h['low'].values
        c_12h = df_12h['close'].values
        
        # Camarilla pivot: P = (H1 + L1 + C1) / 3
        pivot_12h = (h_12h + l_12h + c_12h) / 3.0
        # Range = H1 - L1
        range_12h = h_12h - l_12h
        # Resistance levels: R3 = C1 + (Range * 1.1/4), R4 = C1 + (Range * 1.1/2)
        r3_12h = c_12h + (range_12h * 1.1 / 4)
        r4_12h = c_12h + (range_12h * 1.1 / 2)
        # Support levels: S3 = C1 - (Range * 1.1/4), S4 = C1 - (Range * 1.1/2)
        s3_12h = c_12h - (range_12h * 1.1 / 4)
        s4_12h = c_12h - (range_12h * 1.1 / 2)
        
        # Pivot bias: price above R3 = bullish bias, below S3 = bearish bias
        # We use the aligned values to check if current price is above/below these levels
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
        r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
        r4_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
        s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
        s4_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
        
        # Price above R3 = bullish bias, below S3 = bearish bias
        # Price between S3 and R3 = neutral (no bias)
        price_above_r3 = close > r3_aligned
        price_below_s3 = close < s3_aligned
        # For breakout continuation, we also check if price is above R4 or below S4
        price_above_r4 = close > r4_aligned
        price_below_s4 = close < s4_aligned
    else:
        # Not enough data - neutral bias
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        price_above_r3 = np.full(n, False)
        price_below_s3 = np.full(n, False)
        price_above_r4 = np.full(n, False)
        price_below_s4 = np.full(n, False)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 14, 2)  # Donchian, volume avg, ATR warmup, 12h pivot lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
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
                # 3. Price moves below R3 (pivot bias weakening)
                if price <= stop_price or price <= donchian_low[i] or price < r3_aligned[i]:
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
                # 3. Price moves above S3 (pivot bias weakening)
                if price >= stop_price or price >= donchian_high[i] or price > s3_aligned[i]:
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
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Entry conditions with pivot bias
        # Long: breakout up + volume confirmed + price above R3 (bullish bias)
        # Short: breakout down + volume confirmed + price below S3 (bearish bias)
        if breakout_up and volume_confirmed and price_above_r3[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and price_below_s3[i]:
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