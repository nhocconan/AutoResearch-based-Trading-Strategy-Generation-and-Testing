#!/usr/bin/env python3
"""
Experiment #1611: 6h Camarilla Pivot + 1d Trend Filter + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 1d EMA trend filter and volume confirmation (>1.8x average) captures 
intraday swings with institutional level respect. The 1d EMA (50) filters for 
higher-timeframe trend alignment, reducing false signals in choppy markets. 
Position size 0.25 balances risk and return. Target: 75-175 total trades over 4 years 
(19-44/year) by requiring confluence of pivot level, trend, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1611_6h_camarilla_pivot_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter and pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1d Camarilla Pivot Levels (based on previous day) ===
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #            S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4_1d = typical_price_1d + (range_1d * 1.1 / 2.0)
    r3_1d = typical_price_1d + (range_1d * 1.1 / 4.0)
    s3_1d = typical_price_1d - (range_1d * 1.1 / 4.0)
    s4_1d = typical_price_1d - (range_1d * 1.1 / 2.0)
    
    # Align pivot levels to 6h timeframe (shifted by 1 for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for 1d EMA(50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion or stop after 12 bars ---
        if in_position:
            bars_since_entry += 1
            
            # Time-based exit: max 12 bars (3 days for 6h)
            if bars_since_entry >= 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Mean reversion exit: price reaches opposite pivot level
            if position_side > 0:  # Long position
                if price >= r3_1d_aligned[i]:  # Take profit at R3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price <= s3_1d_aligned[i]:  # Take profit at S3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment
        trend_following = trend_1d_aligned[i] != 0
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if trend_following and volume_spike:
            # Fade at R3/S3 (mean reversion) or breakout at R4/S4
            if price <= r3_1d_aligned[i] and price >= s3_1d_aligned[i]:
                # In the mean reversion zone between S3 and R3
                if price <= s3_1d_aligned[i] * 1.005 and trend_1d_aligned[i] > 0:  # Near S3 in uptrend -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price >= r3_1d_aligned[i] * 0.995 and trend_1d_aligned[i] < 0:  # Near R3 in downtrend -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif price > r4_1d_aligned[i] and trend_1d_aligned[i] > 0:  # Breakout above R4 in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < s4_1d_aligned[i] and trend_1d_aligned[i] < 0:  # Breakdown below S4 in downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1611: 6h Camarilla Pivot + 1d Trend Filter + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 1d EMA trend filter and volume confirmation (>1.8x average) captures 
intraday swings with institutional level respect. The 1d EMA (50) filters for 
higher-timeframe trend alignment, reducing false signals in choppy markets. 
Position size 0.25 balances risk and return. Target: 75-175 total trades over 4 years 
(19-44/year) by requiring confluence of pivot level, trend, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1611_6h_camarilla_pivot_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter and pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1d Camarilla Pivot Levels (based on previous day) ===
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #            S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4_1d = typical_price_1d + (range_1d * 1.1 / 2.0)
    r3_1d = typical_price_1d + (range_1d * 1.1 / 4.0)
    s3_1d = typical_price_1d - (range_1d * 1.1 / 4.0)
    s4_1d = typical_price_1d - (range_1d * 1.1 / 2.0)
    
    # Align pivot levels to 6h timeframe (shifted by 1 for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for 1d EMA(50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion or stop after 12 bars ---
        if in_position:
            bars_since_entry += 1
            
            # Time-based exit: max 12 bars (3 days for 6h)
            if bars_since_entry >= 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Mean reversion exit: price reaches opposite pivot level
            if position_side > 0:  # Long position
                if price >= r3_1d_aligned[i]:  # Take profit at R3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price <= s3_1d_aligned[i]:  # Take profit at S3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment
        trend_following = trend_1d_aligned[i] != 0
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if trend_following and volume_spike:
            # Fade at R3/S3 (mean reversion) or breakout at R4/S4
            if price <= r3_1d_aligned[i] and price >= s3_1d_aligned[i]:
                # In the mean reversion zone between S3 and R3
                if price <= s3_1d_aligned[i] * 1.005 and trend_1d_aligned[i] > 0:  # Near S3 in uptrend -> long
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price >= r3_1d_aligned[i] * 0.995 and trend_1d_aligned[i] < 0:  # Near R3 in downtrend -> short
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif price > r4_1d_aligned[i] and trend_1d_aligned[i] > 0:  # Breakout above R4 in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < s4_1d_aligned[i] and trend_1d_aligned[i] < 0:  # Breakdown below S4 in downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals