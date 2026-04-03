#!/usr/bin/env python3
"""
Experiment #1651: 6h Camarilla Pivot + Volume Spike + 1d Trend Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with volume spikes (>1.5x average) and 1d EMA50 trend filter capture institutional order flow at key levels. Mean reversion at R3/S3 in ranging markets, breakout continuation at R4/S4 in trending markets. Position size 0.25 balances risk and return. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1651_6h_camarilla_pivot_vol_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels from previous day ===
    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Previous day's typical price (using 4x 6h bars = 1 day)
    prev_typical = pd.Series(typical_price).shift(4).values
    # Previous day's high/low
    prev_high = pd.Series(high).shift(4).values
    prev_low = pd.Series(low).shift(4).values
    
    # Camarilla calculations
    camarilla_range = prev_high - prev_low
    # R3, S3 for mean reversion (1.0718 * range from close)
    r3 = prev_typical + camarilla_range * 1.0718
    s3 = prev_typical - camarilla_range * 1.0718
    # R4, S4 for breakout (1.382 * range from close)
    r4 = prev_typical + camarilla_range * 1.382
    s4 = prev_typical - camarilla_range * 1.382
    
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
    
    warmup = 20  # sufficient for volume MA and pivot calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion or stoploss ---
        if in_position:
            if position_side > 0:  # Long position
                # Take profit at S3 (mean reversion target) or stoploss if breaks S4
                if price <= s3[i]:  # TP at S3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                elif price < s4[i]:  # SL if breaks S4 (failed breakout)
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Take profit at R3 (mean reversion target) or stoploss if breaks R4
                if price >= r3[i]:  # TP at R3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                elif price > r4[i]:  # SL if breaks R4 (failed breakout)
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Mean reversion at R3/S3 (fade extreme moves)
            if price >= r3[i] and trend_1d_aligned[i] < 0:  # Short at R3 in downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            elif price <= s3[i] and trend_1d_aligned[i] > 0:  # Long at S3 in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Breakout continuation at R4/S4 (institutional breakout)
            elif price > r4[i] and trend_1d_aligned[i] > 0:  # Long breakout in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif price < s4[i] and trend_1d_aligned[i] < 0:  # Short breakdown in downtrend
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
Experiment #1651: 6h Camarilla Pivot + Volume Spike + 1d Trend Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with volume spikes (>1.5x average) and 1d EMA50 trend filter capture institutional order flow at key levels. Mean reversion at R3/S3 in ranging markets, breakout continuation at R4/S4 in trending markets. Position size 0.25 balances risk and return. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1651_6h_camarilla_pivot_vol_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels from previous day ===
    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Previous day's typical price (using 4x 6h bars = 1 day)
    prev_typical = pd.Series(typical_price).shift(4).values
    # Previous day's high/low
    prev_high = pd.Series(high).shift(4).values
    prev_low = pd.Series(low).shift(4).values
    
    # Camarilla calculations
    camarilla_range = prev_high - prev_low
    # R3, S3 for mean reversion (1.0718 * range from close)
    r3 = prev_typical + camarilla_range * 1.0718
    s3 = prev_typical - camarilla_range * 1.0718
    # R4, S4 for breakout (1.382 * range from close)
    r4 = prev_typical + camarilla_range * 1.382
    s4 = prev_typical - camarilla_range * 1.382
    
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
    
    warmup = 20  # sufficient for volume MA and pivot calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion or stoploss ---
        if in_position:
            if position_side > 0:  # Long position
                # Take profit at S3 (mean reversion target) or stoploss if breaks S4
                if price <= s3[i]:  # TP at S3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                elif price < s4[i]:  # SL if breaks S4 (failed breakout)
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Take profit at R3 (mean reversion target) or stoploss if breaks R4
                if price >= r3[i]:  # TP at R3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                elif price > r4[i]:  # SL if breaks R4 (failed breakout)
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Mean reversion at R3/S3 (fade extreme moves)
            if price >= r3[i] and trend_1d_aligned[i] < 0:  # Short at R3 in downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            elif price <= s3[i] and trend_1d_aligned[i] > 0:  # Long at S3 in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Breakout continuation at R4/S4 (institutional breakout)
            elif price > r4[i] and trend_1d_aligned[i] > 0:  # Long breakout in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif price < s4[i] and trend_1d_aligned[i] < 0:  # Short breakdown in downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals