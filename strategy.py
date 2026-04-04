#!/usr/bin/env python3
"""
Experiment #3455: 6h Camarilla Pivot Fade/Breakout with Weekly Trend Filter
HYPOTHESIS: Camarilla pivot levels from 1d provide high-probability fade zones at R3/S3 and breakout continuation at R4/S4. 
Weekly trend filter (price vs weekly EMA20) ensures alignment with higher timeframe momentum. 
Volume confirmation (>1.5x 20-period average) filters false breakouts. 
Designed for 6h timeframe to capture swing trades in both bull (breakout continuation) and bear (fade at extremes) markets.
Target: 75-150 total trades over 4 years (19-38/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3455_6h_camarilla_pivot_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # PP = (H + L + C) / 3
    # R4 = PP + ((H - L) * 1.1 / 2)
    # R3 = PP + ((H - L) * 1.1 / 4)
    # S3 = PP - ((H - L) * 1.1 / 4)
    # S4 = PP - ((H - L) * 1.1 / 2)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = pp_1d + ((high_1d - low_1d) * 1.1 / 2.0)
    r3_1d = pp_1d + ((high_1d - low_1d) * 1.1 / 4.0)
    s3_1d = pp_1d - ((high_1d - low_1d) * 1.1 / 4.0)
    s4_1d = pp_1d - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h timeframe (shifted by 1 for completed bars)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = max(20, 20)  # sufficient for volume MA and weekly EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion at opposite Camarilla level ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit long when price reaches S3 (mean reversion) or breaks below S4 (stop)
                if price <= s3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price < s4_1d_aligned[i]:  # Stop loss if breaks S4
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit short when price reaches R3 (mean reversion) or breaks above R4 (stop)
                if price >= r3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price > r4_1d_aligned[i]:  # Stop loss if breaks R4
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Weekly trend filter: only trade in direction of weekly EMA20
            weekly_bias = price - ema_1w_aligned[i]
            
            # Fade logic at extreme Camarilla levels (R3/S3)
            # Short fade at R3 when price is above weekly EMA (bearish bias in uptrend)
            if price >= r3_1d_aligned[i] and weekly_bias > 0:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            # Long fade at S3 when price is below weekly EMA (bullish bias in downtrend)
            elif price <= s3_1d_aligned[i] and weekly_bias < 0:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Breakout continuation logic at R4/S4
            # Breakout long when price closes above R4 with bullish weekly bias
            elif price > r4_1d_aligned[i] and weekly_bias > 0:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Breakout short when price closes below S4 with bearish weekly bias
            elif price < s4_1d_aligned[i] and weekly_bias < 0:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals