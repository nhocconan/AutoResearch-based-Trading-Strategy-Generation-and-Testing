#!/usr/bin/env python3
"""
Experiment #171: 6h Camarilla Pivot Fade/Breakout + Volume Spike + ATR Stoploss

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) filtered by 
volume spikes (>2.0x average) and ATR-based stoploss capture mean reversion in ranges 
and momentum in breakouts. Uses discrete position sizing (0.25) to minimize churn. 
HTF = 1d for pivot calculation. Targets 12-37 trades/year (50-150 total over 4 years) 
by requiring confluence of pivot level, volume, and price action. Works in bull (breakouts) 
and bear (fades at resistance) markets. 6h timeframe reduces noise vs lower TFs.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold set to 2.0x to balance signal quality and frequency
- ATR-based stoploss at 2.0*ATR for risk management
- Warmup period set to 100 bars for stable indicators
- Pivot levels calculated from prior 1d OHLC (standard Camarilla formula)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_171_6h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Standard Camarilla: based on previous day's OHLC
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    range_hl = df_1d['high'] - df_1d['low']
    
    # Resistance levels
    r3 = pivot + range_hl * 1.1 / 2.0
    r4 = pivot + range_hl * 1.1
    
    # Support levels
    s3 = pivot - range_hl * 1.1 / 2.0
    s4 = pivot - range_hl * 1.1
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position
    
    warmup = 100  # Warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Conditions ---
        # Fade at R3/S3 (mean reversion)
        fade_short = close[i] > r3_aligned[i] and close[i] < r4_aligned[i]
        fade_long = close[i] < s3_aligned[i] and close[i] > s4_aligned[i]
        
        # Breakout continuation at R4/S4 (momentum)
        breakout_up = close[i] > r4_aligned[i]
        breakout_down = close[i] < s4_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Fade at S3/S4 OR breakout above R4 + volume spike
        long_condition = (fade_long or breakout_up) and volume_spike
        
        # Short: Fade at R3/R4 OR breakout below S4 + volume spike
        short_condition = (fade_short or breakout_down) and volume_spike
        
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
Experiment #171: 6h Camarilla Pivot Fade/Breakout + Volume Spike + ATR Stoploss

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) filtered by 
volume spikes (>2.0x average) and ATR-based stoploss capture mean reversion in ranges 
and momentum in breakouts. Uses discrete position sizing (0.25) to minimize churn. 
HTF = 1d for pivot calculation. Targets 12-37 trades/year (50-150 total over 4 years) 
by requiring confluence of pivot level, volume, and price action. Works in bull (breakouts) 
and bear (fades at resistance) markets. 6h timeframe reduces noise vs lower TFs.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold set to 2.0x to balance signal quality and frequency
- ATR-based stoploss at 2.0*ATR for risk management
- Warmup period set to 100 bars for stable indicators
- Pivot levels calculated from prior 1d OHLC (standard Camarilla formula)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_171_6h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Standard Camarilla: based on previous day's OHLC
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    range_hl = df_1d['high'] - df_1d['low']
    
    # Resistance levels
    r3 = pivot + range_hl * 1.1 / 2.0
    r4 = pivot + range_hl * 1.1
    
    # Support levels
    s3 = pivot - range_hl * 1.1 / 2.0
    s4 = pivot - range_hl * 1.1
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position
    
    warmup = 100  # Warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Conditions ---
        # Fade at R3/S3 (mean reversion)
        fade_short = close[i] > r3_aligned[i] and close[i] < r4_aligned[i]
        fade_long = close[i] < s3_aligned[i] and close[i] > s4_aligned[i]
        
        # Breakout continuation at R4/S4 (momentum)
        breakout_up = close[i] > r4_aligned[i]
        breakout_down = close[i] < s4_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Fade at S3/S4 OR breakout above R4 + volume spike
        long_condition = (fade_long or breakout_up) and volume_spike
        
        # Short: Fade at R3/R4 OR breakout below S4 + volume spike
        short_condition = (fade_short or breakout_down) and volume_spike
        
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