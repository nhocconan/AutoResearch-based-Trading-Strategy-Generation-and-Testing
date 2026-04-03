#!/usr/bin/env python3
"""
Experiment #159: 6h Camarilla Pivot + 12h Volume Spike + ATR Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 12h volume spikes (>2.0x average) and ATR-based trend filter capture 
institutional order flow at key levels. Works in bull markets (breakouts at R4/S4) 
and bear markets (mean reversion at R3/S3). 6h timeframe targets 12-37 trades/year 
(50-150 total over 4 years) to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_159_6h_camarilla_12h_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from previous 12h bar
    pivots_high = np.full(n, np.nan)
    pivots_low = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous completed 12h bar for pivot calculation
        ph = df_12h['high'].values[i-1]
        pl = df_12h['low'].values[i-1]
        pc = df_12h['close'].values[i-1]
        
        if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
            pivots_high[i] = ph
            pivots_low[i] = pl
            # Camarilla levels
            range_val = ph - pl
            r3[i] = pc + range_val * 1.1 / 4
            s3[i] = pc - range_val * 1.1 / 4
            r4[i] = pc + range_val * 1.1 / 2
            s4[i] = pc - range_val * 1.1 / 2
    
    # === HTF: 1d data for volume spike detection (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).apply(
        lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1.0, raw=False
    ).values
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h Indicators: ATR(14) for trend and stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x 1d average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- ATR Trend Filter: ATR > MA(50) indicates high volatility/trending market ---
        atr_expanding = atr_14[i] > atr_ma_50[i]
        
        # --- Camarilla Level Conditions ---
        # Mean reversion at R3/S3 (price extreme likely to revert)
        mean_revert_long = close[i] <= s3[i] and close[i] > s4[i]  # Between S3 and S4
        mean_revert_short = close[i] >= r3[i] and close[i] < r4[i]  # Between R3 and R4
        
        # Breakout continuation at R4/S4 (price breaks extreme level)
        breakout_long = close[i] > r4[i]
        breakout_short = close[i] < s4[i]
        
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
                # Exit on mean reversion signal opposite direction
                if position_side > 0 and mean_revert_short:
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
                # Exit on mean reversion signal opposite direction
                if position_side < 0 and mean_revert_long:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Mean reversion at S3/S4 OR breakout at R4, with volume spike and ATR expansion
        long_condition = (volume_spike and atr_expanding and 
                         (mean_revert_long or breakout_long))
        
        # Short: Mean reversion at R3/R4 OR breakout at S4, with volume spike and ATR expansion
        short_condition = (volume_spike and atr_expanding and 
                          (mean_revert_short or breakout_short))
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #159: 6h Camarilla Pivot + 12h Volume Spike + ATR Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 12h volume spikes (>2.0x average) and ATR-based trend filter capture 
institutional order flow at key levels. Works in bull markets (breakouts at R4/S4) 
and bear markets (mean reversion at R3/S3). 6h timeframe targets 12-37 trades/year 
(50-150 total over 4 years) to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_159_6h_camarilla_12h_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from previous 12h bar
    pivots_high = np.full(n, np.nan)
    pivots_low = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous completed 12h bar for pivot calculation
        ph = df_12h['high'].values[i-1]
        pl = df_12h['low'].values[i-1]
        pc = df_12h['close'].values[i-1]
        
        if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
            pivots_high[i] = ph
            pivots_low[i] = pl
            # Camarilla levels
            range_val = ph - pl
            r3[i] = pc + range_val * 1.1 / 4
            s3[i] = pc - range_val * 1.1 / 4
            r4[i] = pc + range_val * 1.1 / 2
            s4[i] = pc - range_val * 1.1 / 2
    
    # === HTF: 1d data for volume spike detection (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).apply(
        lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1.0, raw=False
    ).values
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h Indicators: ATR(14) for trend and stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x 1d average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- ATR Trend Filter: ATR > MA(50) indicates high volatility/trending market ---
        atr_expanding = atr_14[i] > atr_ma_50[i]
        
        # --- Camarilla Level Conditions ---
        # Mean reversion at R3/S3 (price extreme likely to revert)
        mean_revert_long = close[i] <= s3[i] and close[i] > s4[i]  # Between S3 and S4
        mean_revert_short = close[i] >= r3[i] and close[i] < r4[i]  # Between R3 and R4
        
        # Breakout continuation at R4/S4 (price breaks extreme level)
        breakout_long = close[i] > r4[i]
        breakout_short = close[i] < s4[i]
        
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
                # Exit on mean reversion signal opposite direction
                if position_side > 0 and mean_revert_short:
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
                # Exit on mean reversion signal opposite direction
                if position_side < 0 and mean_revert_long:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Mean reversion at S3/S4 OR breakout at R4, with volume spike and ATR expansion
        long_condition = (volume_spike and atr_expanding and 
                         (mean_revert_long or breakout_long))
        
        # Short: Mean reversion at R3/R4 OR breakout at S4, with volume spike and ATR expansion
        short_condition = (volume_spike and atr_expanding and 
                          (mean_revert_short or breakout_short))
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>