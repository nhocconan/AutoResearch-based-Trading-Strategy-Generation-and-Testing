#!/usr/bin/env python3
"""
Experiment #175: 6h Williams %R Reversal + 1w Trend Filter + Volume Spike

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h.
Trades are taken only in direction of 1w EMA(50) trend to avoid counter-trend whipsaws.
Volume confirmation (>1.5x 20-period average volume on 1d) ensures breakout validity.
ATR-based stoploss (2.0*ATR) manages risk. Target: 75-150 total trades over 4 years.
Works in bull/bear markets by trading reversals within the dominant weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === HTF: 1d data for volume spike filter ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_vol_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Reversal Signals ---
        # Oversold: Williams %R crosses above -80 from below
        oversold = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
        # Overbought: Williams %R crosses below -20 from above
        overbought = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
        
        # --- Position Management ---
        if in_position:
            # Check stoploss: 2.0 * ATR against position
            if position_side > 0:  # Long
                if close[i] < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if close[i] > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Still in position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: oversold reversal + weekly uptrend + volume confirmation
        if oversold and (close[i] > ema_1w_aligned[i]) and vol_spike_1d_aligned[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: overbought reversal + weekly downtrend + volume confirmation
        elif overbought and (close[i] < ema_1w_aligned[i]) and vol_spike_1d_aligned[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #175: 6h Williams %R Reversal + 1w Trend Filter + Volume Spike

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h.
Trades are taken only in direction of 1w EMA(50) trend to avoid counter-trend whipsaws.
Volume confirmation (>1.5x 20-period average volume on 1d) ensures breakout validity.
ATR-based stoploss (2.0*ATR) manages risk. Target: 75-150 total trades over 4 years.
Works in bull/bear markets by trading reversals within the dominant weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === HTF: 1d data for volume spike filter ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_vol_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Reversal Signals ---
        # Oversold: Williams %R crosses above -80 from below
        oversold = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
        # Overbought: Williams %R crosses below -20 from above
        overbought = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
        
        # --- Position Management ---
        if in_position:
            # Check stoploss: 2.0 * ATR against position
            if position_side > 0:  # Long
                if close[i] < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if close[i] > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Still in position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: oversold reversal + weekly uptrend + volume confirmation
        if oversold and (close[i] > ema_1w_aligned[i]) and vol_spike_1d_aligned[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: overbought reversal + weekly downtrend + volume confirmation
        elif overbought and (close[i] < ema_1w_aligned[i]) and vol_spike_1d_aligned[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals

</think>