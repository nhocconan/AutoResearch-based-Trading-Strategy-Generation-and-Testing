#!/usr/bin/env python3
"""
Experiment #1695: 6h Williams %R + 1w EMA Trend + Volume Confirmation
HYPOTHESIS: 6h Williams %R (oversold < -80, overbought > -20) with 1w EMA trend alignment and volume confirmation (>1.3x average) captures mean reversion in ranging markets and continuation in trending markets. The 1w EMA filter ensures trades align with the weekly trend, reducing false signals during strong moves. Position size fixed at 0.25 to manage drawdown. Target: 75-150 total trades over 4 years (19-37/year) by using moderate entry conditions and multi-timeframe confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1695_6h_williamsr_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # EMA(21) for weekly trend
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, -50.0)  # default neutral
    # Avoid division by zero
    denom = highest_high - lowest_low
    mask = denom != 0
    williams_r[mask] = -100 * (highest_high[mask] - close[mask]) / denom[mask]
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 20  # sufficient for Williams %R and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or extreme reversal ---
        if in_position:
            # Exit conditions: reverse Williams %R signal or extreme opposite reading
            if position_side > 0:  # Long position
                # Exit if Williams %R goes above -20 (overbought) or strongly negative reversal
                if williams_r[i] > -20 or (williams_r[i] < -80 and trend_1w_aligned[i] < 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit if Williams %R goes below -80 (oversold) or strongly positive reversal
                if williams_r[i] < -80 or (williams_r[i] > -20 and trend_1w_aligned[i] > 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1w trend alignment
        trend_following = trend_1w_aligned[i] != 0  # Should always be ±1
        
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if trend_following and volume_spike:
            # Mean reversion: buy oversold in uptrend, sell overbought in downtrend
            if williams_r[i] < -80 and trend_1w_aligned[i] > 0:  # Oversold in uptrend -> long
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif williams_r[i] > -20 and trend_1w_aligned[i] < 0:  # Overbought in downtrend -> short
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1695: 6h Williams %R + 1w EMA Trend + Volume Confirmation
HYPOTHESIS: 6h Williams %R (oversold < -80, overbought > -20) with 1w EMA trend alignment and volume confirmation (>1.3x average) captures mean reversion in ranging markets and continuation in trending markets. The 1w EMA filter ensures trades align with the weekly trend, reducing false signals during strong moves. Position size fixed at 0.25 to manage drawdown. Target: 75-150 total trades over 4 years (19-37/year) by using moderate entry conditions and multi-timeframe confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1695_6h_williamsr_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # EMA(21) for weekly trend
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, -50.0)  # default neutral
    # Avoid division by zero
    denom = highest_high - lowest_low
    mask = denom != 0
    williams_r[mask] = -100 * (highest_high[mask] - close[mask]) / denom[mask]
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 20  # sufficient for Williams %R and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or extreme reversal ---
        if in_position:
            # Exit conditions: reverse Williams %R signal or extreme opposite reading
            if position_side > 0:  # Long position
                # Exit if Williams %R goes above -20 (overbought) or strongly negative reversal
                if williams_r[i] > -20 or (williams_r[i] < -80 and trend_1w_aligned[i] < 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit if Williams %R goes below -80 (oversold) or strongly positive reversal
                if williams_r[i] < -80 or (williams_r[i] > -20 and trend_1w_aligned[i] > 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1w trend alignment
        trend_following = trend_1w_aligned[i] != 0  # Should always be ±1
        
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if trend_following and volume_spike:
            # Mean reversion: buy oversold in uptrend, sell overbought in downtrend
            if williams_r[i] < -80 and trend_1w_aligned[i] > 0:  # Oversold in uptrend -> long
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif williams_r[i] > -20 and trend_1w_aligned[i] < 0:  # Overbought in downtrend -> short
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals