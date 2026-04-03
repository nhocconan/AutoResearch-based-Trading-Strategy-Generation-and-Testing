#!/usr/bin/env python3
"""
Experiment #032: 12h Donchian Breakout + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and aligned with 1w trend (price > EMA50 for long, < EMA50 for short), captures high-probability 
trend continuation moves. The 12h timeframe minimizes fee drag while Donchian breakouts provide 
objective entry/exit levels. Volume confirms institutional participation, and the 1w trend filter 
ensures we only trade in the direction of the higher timeframe trend, reducing false signals in 
choppy markets. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
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
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 20:
            lookback_high = high[i-20:i]
            lookback_low = low[i-20:i]
            donchian_high[i] = np.max(lookback_high)
            donchian_low[i] = np.min(lookback_low)
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1w EMA50 ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
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
                # Exit when price touches opposite Donchian band (mean reversion) or at Donchian low for breakout
                if close[i] <= donchian_low[i]:
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
                # Exit when price touches opposite Donchian band (mean reversion) or at Donchian high for breakdown
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_1w_ema
        )
        
        # Short: Price breaks below Donchian low with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_1w_ema
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
Experiment #032: 12h Donchian Breakout + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and aligned with 1w trend (price > EMA50 for long, < EMA50 for short), captures high-probability 
trend continuation moves. The 12h timeframe minimizes fee drag while Donchian breakouts provide 
objective entry/exit levels. Volume confirms institutional participation, and the 1w trend filter 
ensures we only trade in the direction of the higher timeframe trend, reducing false signals in 
choppy markets. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
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
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 20:
            lookback_high = high[i-20:i]
            lookback_low = low[i-20:i]
            donchian_high[i] = np.max(lookback_high)
            donchian_low[i] = np.min(lookback_low)
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1w EMA50 ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
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
                # Exit when price touches opposite Donchian band (mean reversion) or at Donchian low for breakout
                if close[i] <= donchian_low[i]:
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
                # Exit when price touches opposite Donchian band (mean reversion) or at Donchian high for breakdown
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_1w_ema
        )
        
        # Short: Price breaks below Donchian low with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_1w_ema
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