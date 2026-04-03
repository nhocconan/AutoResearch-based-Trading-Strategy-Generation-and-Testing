#!/usr/bin/env python3
"""
Experiment #150: 1d Donchian(20) Breakout + Weekly Volume Spike + Weekly Pivot Direction

HYPOTHESIS: Donchian(20) breakouts on 1d timeframe with volume confirmation (>2x 20-period weekly average volume) 
and weekly pivot direction filter (price above/below weekly pivot) captures strong momentum moves. 
Weekly pivot provides structural bias: long only when price > weekly pivot, short only when price < weekly pivot. 
This avoids counter-trend breakouts that fail in ranging/bear markets. Using 1d timeframe targets 30-100 trades 
over 4 years (7-25/year) to minimize fee drag. Weekly timeframe HTF filters ensure alignment with longer-term 
structure, working in both bull and bear markets via adaptive pivot bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_volume_weekly_pivot_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume spike filter and weekly pivot ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly volume spike: >2x 20-period average volume
    avg_vol_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_vol_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Weekly pivot calculation from prior week's OHLC
    # Using current week's high/low/close for pivot (standard formula)
    weekly_pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1w_aligned[i]) or np.isnan(weekly_pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Direction Filter ---
        bullish_bias = close[i] > weekly_pivot_1w_aligned[i]   # Price above weekly pivot
        bearish_bias = close[i] < weekly_pivot_1w_aligned[i]   # Price below weekly pivot
        
        # --- Donchian Breakout + Volume Confirmation ---
        # Upper breakout: price breaks above Donchian high with volume spike
        upper_breakout = (close[i] > donchian_high[i]) and vol_spike_1w_aligned[i]
        # Lower breakout: price breaks below Donchian low with volume spike
        lower_breakout = (close[i] < donchian_low[i]) and vol_spike_1w_aligned[i]
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit when price returns to Donchian midpoint (mean reversion within channel)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if position_side > 0:  # Long
                if close[i] < donchian_mid:
                    in_position = False
                    position_side = 0
            else:  # Short
                if close[i] > donchian_mid:
                    in_position = False
                    position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: upper breakout + bullish bias (price above weekly pivot)
        if upper_breakout and bullish_bias:
            in_position = True
            position_side = 1
            signals[i] = SIZE
        # Short: lower breakout + bearish bias (price below weekly pivot)
        elif lower_breakout and bearish_bias:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #150: 1d Donchian(20) Breakout + Weekly Volume Spike + Weekly Pivot Direction

HYPOTHESIS: Donchian(20) breakouts on 1d timeframe with volume confirmation (>2x 20-period weekly average volume) 
and weekly pivot direction filter (price above/below weekly pivot) captures strong momentum moves. 
Weekly pivot provides structural bias: long only when price > weekly pivot, short only when price < weekly pivot. 
This avoids counter-trend breakouts that fail in ranging/bear markets. Using 1d timeframe targets 30-100 trades 
over 4 years (7-25/year) to minimize fee drag. Weekly timeframe HTF filters ensure alignment with longer-term 
structure, working in both bull and bear markets via adaptive pivot bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_volume_weekly_pivot_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume spike filter and weekly pivot ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly volume spike: >2x 20-period average volume
    avg_vol_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_vol_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Weekly pivot calculation from prior week's OHLC
    # Using current week's high/low/close for pivot (standard formula)
    weekly_pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1w_aligned[i]) or np.isnan(weekly_pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Direction Filter ---
        bullish_bias = close[i] > weekly_pivot_1w_aligned[i]   # Price above weekly pivot
        bearish_bias = close[i] < weekly_pivot_1w_aligned[i]   # Price below weekly pivot
        
        # --- Donchian Breakout + Volume Confirmation ---
        # Upper breakout: price breaks above Donchian high with volume spike
        upper_breakout = (close[i] > donchian_high[i]) and vol_spike_1w_aligned[i]
        # Lower breakout: price breaks below Donchian low with volume spike
        lower_breakout = (close[i] < donchian_low[i]) and vol_spike_1w_aligned[i]
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit when price returns to Donchian midpoint (mean reversion within channel)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if position_side > 0:  # Long
                if close[i] < donchian_mid:
                    in_position = False
                    position_side = 0
            else:  # Short
                if close[i] > donchian_mid:
                    in_position = False
                    position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: upper breakout + bullish bias (price above weekly pivot)
        if upper_breakout and bullish_bias:
            in_position = True
            position_side = 1
            signals[i] = SIZE
        # Short: lower breakout + bearish bias (price below weekly pivot)
        elif lower_breakout and bearish_bias:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
    
    return signals

</think>