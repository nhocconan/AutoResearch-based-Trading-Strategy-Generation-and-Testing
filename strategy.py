#!/usr/bin/env python3
"""
Experiment #5535: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.4x average and aligned with 
weekly pivot structure (price above weekly PP = bullish bias, below = bearish bias) capture 
high-probability moves. The weekly pivot provides multi-week institutional reference that works 
across regimes, while volume confirmation filters false breakouts. Discrete position sizing 
(0.25) and ATR-based trailing stop control risk. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5535_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate weekly pivot levels from previous week's OHLC
        prev_close = df_1w['close'].shift(1).values
        prev_high = df_1w['high'].shift(1).values
        prev_low = df_1w['low'].shift(1).values
        
        # Weekly pivot point (PP) = (H + L + C) / 3
        pp = (prev_high + prev_low + prev_close) / 3.0
        # Weekly range = H - L
        rang = prev_high - prev_low
        
        # Weekly support/resistance levels (standard pivot)
        # R1 = PP + (H-L)
        # S1 = PP - (H-L)
        # R2 = PP + 2*(H-L)
        # S2 = PP - 2*(H-L)
        r1 = pp + rang
        s1 = pp - rang
        r2 = pp + 2 * rang
        s2 = pp - 2 * rang
        
        # Align to LTF (6h) with shift(1) for completed bars only
        pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
        r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for trailing stop ===
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
    highest_since_entry = 0.0  # For long positions
    lowest_since_entry = 0.0   # For short positions
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume avg, ATR warmup
    
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
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or pivot failure ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry (trailing stop)
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price closes below weekly S1 (pivot support fails)
                if price <= stop_price or price <= donchian_low[i] or price < s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry (trailing stop)
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price closes above weekly R1 (pivot resistance fails)
                if price >= stop_price or price >= donchian_high[i] or price > r1_aligned[i]:
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
        
        # Volume confirmation: current volume > 1.4x average volume
        volume_confirmed = volume_ratio[i] > 1.4
        
        # Weekly pivot-based entry conditions:
        # Long: breakout above weekly R2 (strong bullish breakout) OR bounce from S1 (mean reversion in range)
        # Short: breakdown below weekly S2 (strong bearish breakdown) OR bounce from R1 (mean reversion in range)
        long_breakout = breakout_up and price > r2_aligned[i-1]
        long_bounce = price > s1_aligned[i] and low[i] <= s1_aligned[i]  # Touched/bounced off S1
        short_breakout = breakout_down and price < s2_aligned[i-1]
        short_bounce = price < r1_aligned[i] and high[i] >= r1_aligned[i]  # Touched/bounced off R1
        
        # Entry conditions: breakout/bounce + volume confirmation
        if (long_breakout or long_bounce) and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif (short_breakout or short_bounce) and volume_confirmed:
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
Experiment #5535: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.4x average and aligned with 
weekly pivot structure (price above weekly PP = bullish bias, below = bearish bias) capture 
high-probability moves. The weekly pivot provides multi-week institutional reference that works 
across regimes, while volume confirmation filters false breakouts. Discrete position sizing 
(0.25) and ATR-based trailing stop control risk. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5535_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate weekly pivot levels from previous week's OHLC
        prev_close = df_1w['close'].shift(1).values
        prev_high = df_1w['high'].shift(1).values
        prev_low = df_1w['low'].shift(1).values
        
        # Weekly pivot point (PP) = (H + L + C) / 3
        pp = (prev_high + prev_low + prev_close) / 3.0
        # Weekly range = H - L
        rang = prev_high - prev_low
        
        # Weekly support/resistance levels (standard pivot)
        # R1 = PP + (H-L)
        # S1 = PP - (H-L)
        # R2 = PP + 2*(H-L)
        # S2 = PP - 2*(H-L)
        r1 = pp + rang
        s1 = pp - rang
        r2 = pp + 2 * rang
        s2 = pp - 2 * rang
        
        # Align to LTF (6h) with shift(1) for completed bars only
        pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
        r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for trailing stop ===
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
    highest_since_entry = 0.0  # For long positions
    lowest_since_entry = 0.0   # For short positions
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume avg, ATR warmup
    
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
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or pivot failure ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry (trailing stop)
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price closes below weekly S1 (pivot support fails)
                if price <= stop_price or price <= donchian_low[i] or price < s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry (trailing stop)
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price closes above weekly R1 (pivot resistance fails)
                if price >= stop_price or price >= donchian_high[i] or price > r1_aligned[i]:
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
        
        # Volume confirmation: current volume > 1.4x average volume
        volume_confirmed = volume_ratio[i] > 1.4
        
        # Weekly pivot-based entry conditions:
        # Long: breakout above weekly R2 (strong bullish breakout) OR bounce from S1 (mean reversion in range)
        # Short: breakdown below weekly S2 (strong bearish breakdown) OR bounce from R1 (mean reversion in range)
        long_breakout = breakout_up and price > r2_aligned[i-1]
        long_bounce = price > s1_aligned[i] and low[i] <= s1_aligned[i]  # Touched/bounced off S1
        short_breakout = breakout_down and price < s2_aligned[i-1]
        short_bounce = price < r1_aligned[i] and high[i] >= r1_aligned[i]  # Touched/bounced off R1
        
        # Entry conditions: breakout/bounce + volume confirmation
        if (long_breakout or long_bounce) and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif (short_breakout or short_bounce) and volume_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals