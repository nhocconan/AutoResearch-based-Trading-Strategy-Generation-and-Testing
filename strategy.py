#!/usr/bin/env python3
"""
Experiment #076: 12h Donchian Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by ATR-based regime (trending when ATR(14) > ATR(50)), captures strong momentum 
moves in both bull and bear markets. The 12h timeframe minimizes fee drag while Donchian 
breakouts provide objective entry/exit levels. Volume spike confirms institutional 
participation, and ATR regime filter avoids choppy markets. Targets 12-37 trades/year on 
12h timeframe (50-150 total over 4 years) to minimize fee drag while targeting high-probability 
breakouts with 2.5x ATR stoploss.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for ATR regime filter (Call ONCE before loop) ===
    if len(df_1d) >= 50:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate True Range and ATR(14) and ATR(50) on 1d
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
        atr_14_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_50_1d = pd.Series(tr_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        
        # Regime: trending when ATR(14) > ATR(50) (increasing volatility)
        atr_regime = atr_14_1d > atr_50_1d
        atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    else:
        atr_regime_aligned = np.full(n, False)
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h
    # We need to map each 12h bar to the prior 20 12h bars' high/low
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_base = np.full(n, np.nan)  # Midpoint for exit
    
    # For each 12h bar, get the prior 20 12h bars' high/low
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 12h bars before current 12h bar (need 20 bars)
        prior_12h_bars = df_1d[df_1d['open_time'] < current_time]  # Using 1d as proxy for prior periods
        # Actually, we need to use 12h data for Donchian calculation
        df_12h = get_htf_data(prices, '12h')  # This is inefficient but needed for accuracy
        if len(df_12h) >= 20:
            prior_12h_bars = df_12h[df_12h['open_time'] < current_time]
            if len(prior_12h_bars) >= 20:
                recent_20 = prior_12h_bars.tail(20)
                donchian_high[i] = recent_20['high'].max()
                donchian_low[i] = recent_20['low'].min()
                donchian_base[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Clean up - reload 12h data properly for efficiency
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # Vectorized Donchian calculation
        donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        donchian_base_12h = (donchian_high_12h + donchian_low_12h) / 2
        
        # Align to 12h timeframe
        donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
        donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
        donchian_base_aligned = align_htf_to_ltf(prices, df_12h, donchian_base_12h)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
        donchian_base_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_base_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss on 12h timeframe
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(
                    high[j] - low[j],
                    abs(high[j] - close[j-1]),
                    abs(low[j] - close[j-1])
                )
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian base (mean reversion) or opposite band
                if close[i] <= donchian_base_aligned[i] or close[i] >= donchian_high_aligned[i]:
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
                # Take profit at Donchian base (mean reversion) or opposite band
                if close[i] >= donchian_base_aligned[i] or close[i] <= donchian_low_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume spike and ATR regime
        long_condition = (
            close[i] > donchian_high_aligned[i] and 
            vol_ratio_1d_aligned[i] > 2.0 and 
            atr_regime_aligned[i]
        )
        
        # Short: Price breaks below Donchian low with volume spike and ATR regime
        short_condition = (
            close[i] < donchian_low_aligned[i] and 
            vol_ratio_1d_aligned[i] > 2.0 and 
            atr_regime_aligned[i]
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #076: 12h Donchian Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by ATR-based regime (trending when ATR(14) > ATR(50)), captures strong momentum 
moves in both bull and bear markets. The 12h timeframe minimizes fee drag while Donchian 
breakouts provide objective entry/exit levels. Volume spike confirms institutional 
participation, and ATR regime filter avoids choppy markets. Targets 12-37 trades/year on 
12h timeframe (50-150 total over 4 years) to minimize fee drag while targeting high-probability 
breakouts with 2.5x ATR stoploss.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for ATR regime filter (Call ONCE before loop) ===
    if len(df_1d) >= 50:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate True Range and ATR(14) and ATR(50) on 1d
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
        atr_14_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_50_1d = pd.Series(tr_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        
        # Regime: trending when ATR(14) > ATR(50) (increasing volatility)
        atr_regime = atr_14_1d > atr_50_1d
        atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    else:
        atr_regime_aligned = np.full(n, False)
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # Vectorized Donchian calculation
        donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        donchian_base_12h = (donchian_high_12h + donchian_low_12h) / 2
        
        # Align to 12h timeframe
        donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
        donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
        donchian_base_aligned = align_htf_to_ltf(prices, df_12h, donchian_base_12h)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
        donchian_base_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_base_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss on 12h timeframe
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(
                    high[j] - low[j],
                    abs(high[j] - close[j-1]),
                    abs(low[j] - close[j-1])
                )
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian base (mean reversion) or opposite band
                if close[i] <= donchian_base_aligned[i] or close[i] >= donchian_high_aligned[i]:
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
                # Take profit at Donchian base (mean reversion) or opposite band
                if close[i] >= donchian_base_aligned[i] or close[i] <= donchian_low_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume spike and ATR regime
        long_condition = (
            close[i] > donchian_high_aligned[i] and 
            vol_ratio_1d_aligned[i] > 2.0 and 
            atr_regime_aligned[i]
        )
        
        # Short: Price breaks below Donchian low with volume spike and ATR regime
        short_condition = (
            close[i] < donchian_low_aligned[i] and 
            vol_ratio_1d_aligned[i] > 2.0 and 
            atr_regime_aligned[i]
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals