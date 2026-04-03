#!/usr/bin/env python3
"""
Experiment #259: 6h Williams %R + 12h ADX Trend + 1d Volume Spike Filter

HYPOTHESIS: Williams %R identifies overextended conditions on 6h, 12h ADX > 25 filters for trending markets to avoid false signals in ranges, and 1d volume spike (volume > 1.5x 20-period MA) confirms institutional participation. This combination works in both bull and bear markets by only taking mean-reversion trades in strong trends with volume confirmation. Targets 12-30 trades/year (50-120 total over 4 years) to minimize fee drag while capturing high-probability reversals at trend extremes with trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX(14) on 12h data
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr_12h = np.zeros(len(close_12h))
        tr_12h[0] = high_12h[0] - low_12h[0]
        for i in range(1, len(close_12h)):
            tr_12h[i] = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]), abs(low_12h[i] - close_12h[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros(len(close_12h))
        dm_minus = np.zeros(len(close_12h))
        for i in range(1, len(close_12h)):
            up_move = high_12h[i] - high_12h[i-1]
            down_move = low_12h[i-1] - low_12h[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM-
        tr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Volume Spike: volume > 1.5x 20-period MA
    if len(df_1d) >= 20:
        volume_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        vol_spike = volume_1d > (1.5 * vol_ma_20)
        vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    else:
        vol_spike_aligned = np.zeros(n)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    valid = (highest_high_14 != lowest_low_14) & ~(np.isnan(highest_high_14) | np.isnan(lowest_low_14))
    williams_r[valid] = -100 * (highest_high_14[valid] - close[valid]) / (highest_high_14[valid] - lowest_low_14[valid])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (ADX > 25) ---
        if adx_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # --- Volume Spike Filter: Require volume confirmation from 1d ---
        if i < len(vol_spike_aligned) and vol_spike_aligned[i] < 0.5:
            signals[i] = 0.0
            continue
        
        # --- Williams %R Signals ---
        # Oversold: Williams %R < -80 (potential bounce)
        # Overbought: Williams %R > -20 (potential pullback)
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Williams %R > -50 (mean reversion halfway)
                if williams_r[i] > -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Williams %R < -50 (mean reversion halfway)
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Oversold Williams %R with volume spike
        if oversold:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Overbought Williams %R with volume spike
        elif overbought:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #259: 6h Williams %R + 12h ADX Trend + 1d Volume Spike Filter

HYPOTHESIS: Williams %R identifies overextended conditions on 6h, 12h ADX > 25 filters for trending markets to avoid false signals in ranges, and 1d volume spike (volume > 1.5x 20-period MA) confirms institutional participation. This combination works in both bull and bear markets by only taking mean-reversion trades in strong trends with volume confirmation. Targets 12-30 trades/year (50-120 total over 4 years) to minimize fee drag while capturing high-probability reversals at trend extremes with trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX(14) on 12h data
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr_12h = np.zeros(len(close_12h))
        tr_12h[0] = high_12h[0] - low_12h[0]
        for i in range(1, len(close_12h)):
            tr_12h[i] = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]), abs(low_12h[i] - close_12h[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros(len(close_12h))
        dm_minus = np.zeros(len(close_12h))
        for i in range(1, len(close_12h)):
            up_move = high_12h[i] - high_12h[i-1]
            down_move = low_12h[i-1] - low_12h[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM-
        tr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Volume Spike: volume > 1.5x 20-period MA
    if len(df_1d) >= 20:
        volume_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        vol_spike = volume_1d > (1.5 * vol_ma_20)
        vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    else:
        vol_spike_aligned = np.zeros(n)
    
    # === 6h Indicators ===
    # Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    valid = (highest_high_14 != lowest_low_14) & ~(np.isnan(highest_high_14) | np.isnan(lowest_low_14))
    williams_r[valid] = -100 * (highest_high_14[valid] - close[valid]) / (highest_high_14[valid] - lowest_low_14[valid])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (ADX > 25) ---
        if adx_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # --- Volume Spike Filter: Require volume confirmation from 1d ---
        if i < len(vol_spike_aligned) and vol_spike_aligned[i] < 0.5:
            signals[i] = 0.0
            continue
        
        # --- Williams %R Signals ---
        # Oversold: Williams %R < -80 (potential bounce)
        # Overbought: Williams %R > -20 (potential pullback)
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Williams %R > -50 (mean reversion halfway)
                if williams_r[i] > -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Williams %R < -50 (mean reversion halfway)
                if williams_r[i] < -50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Oversold Williams %R with volume spike
        if oversold:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Overbought Williams %R with volume spike
        elif overbought:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals