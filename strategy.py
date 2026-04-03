#!/usr/bin/env python3
"""
Experiment #079: 6h Elder Ray + 1d Trend Filter + Volume Spike
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA13.
In strong trends (ADX>25 on 1d), Elder Ray extremes with volume confirmation (>2x average) indicate
continuation signals. In ranging markets (ADX<20), fade at Elder Ray extremes. This adapts to
bull/bear regimes by using 1d ADX to determine market state and applying appropriate logic.
Target: 75-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_079_6h_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for regime detection ===
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    adx_1d = calculate_adx(h_1d, l_1d, c_1d, 14)
    adx_1d_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    bull_power = high - ema13
    bear_power = low - ema13
    
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
    bars_since_entry = 0
    
    warmup = 50  # Warmup for EMA13 and volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_1d_6h[i]) or np.isnan(ema13[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 2.0  # Volume spike threshold
        
        # --- Regime Detection (from 1d ADX) ---
        is_trending = adx_1d_6h[i] > 25
        is_ranging = adx_1d_6h[i] < 20
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions based on regime
            if is_trending:
                # In trending market: exit on Elder Ray divergence with volume
                if position_side > 0:  # Long
                    if bear_power[i] > -bull_power[i] * 0.5 and vol_spike:  # Bear power weakening
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if bull_power[i] < -bear_power[i] * 0.5 and vol_spike:  # Bull power weakening
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Ranging market
                # Exit when Elder Power returns to zero or opposite extreme
                if position_side > 0:  # Long
                    if bull_power[i] < 0:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if bear_power[i] > 0:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if is_trending:
            # Trending market: Elder Ray extremes with volume confirmation
            if bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 80) and vol_spike:
                # Strong bull power - go long
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 20) and vol_spike:
                # Strong bear power - go short
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif is_ranging:
            # Ranging market: fade at Elder Ray extremes
            if bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 80) and vol_spike:
                # Extreme bull power - short fade
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 20) and vol_spike:
                # Extreme bear power - long fade
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            # Transition regime (ADX between 20-25) - no trade
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #079: 6h Elder Ray + 1d Trend Filter + Volume Spike
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA13.
In strong trends (ADX>25 on 1d), Elder Ray extremes with volume confirmation (>2x average) indicate
continuation signals. In ranging markets (ADX<20), fade at Elder Ray extremes. This adapts to
bull/bear regimes by using 1d ADX to determine market state and applying appropriate logic.
Target: 75-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_079_6h_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for regime detection ===
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    adx_1d = calculate_adx(h_1d, l_1d, c_1d, 14)
    adx_1d_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    bull_power = high - ema13
    bear_power = low - ema13
    
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
    bars_since_entry = 0
    
    warmup = 50  # Warmup for EMA13 and volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_1d_6h[i]) or np.isnan(ema13[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 2.0  # Volume spike threshold
        
        # --- Regime Detection (from 1d ADX) ---
        is_trending = adx_1d_6h[i] > 25
        is_ranging = adx_1d_6h[i] < 20
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions based on regime
            if is_trending:
                # In trending market: exit on Elder Ray divergence with volume
                if position_side > 0:  # Long
                    if bear_power[i] > -bull_power[i] * 0.5 and vol_spike:  # Bear power weakening
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if bull_power[i] < -bear_power[i] * 0.5 and vol_spike:  # Bull power weakening
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Ranging market
                # Exit when Elder Power returns to zero or opposite extreme
                if position_side > 0:  # Long
                    if bull_power[i] < 0:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
                else:  # Short
                    if bear_power[i] > 0:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if is_trending:
            # Trending market: Elder Ray extremes with volume confirmation
            if bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 80) and vol_spike:
                # Strong bull power - go long
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 20) and vol_spike:
                # Strong bear power - go short
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif is_ranging:
            # Ranging market: fade at Elder Ray extremes
            if bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 80) and vol_spike:
                # Extreme bull power - short fade
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 20) and vol_spike:
                # Extreme bear power - long fade
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            # Transition regime (ADX between 20-25) - no trade
            signals[i] = 0.0
    
    return signals