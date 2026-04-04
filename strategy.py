#!/usr/bin/env python3
"""
Experiment #3115: 6h Camarilla Pivot + Weekly Trend Filter + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with weekly trend filter (price vs weekly EMA20) and volume confirmation 
creates high-probability entries. In bull markets, buy R3 bounces and break R4; 
in bear markets, sell R3 rallies and break S4. Weekly trend ensures alignment with 
higher timeframe momentum. Volume spike (>1.5x 20-period average) filters weak moves. 
Position size 0.25. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3115_6h_camarilla_pivot_weekly_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA(20) on weekly close
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === HTF: Daily data for Camarilla pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + (high_1d - low_1d) * 1.1 / 2.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    s4 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
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
    
    warmup = max(20, 20)  # sufficient for volume MA and weekly EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or stoploss ---
        if in_position:
            # Exit conditions: reverse signal or price moves against position
            if position_side > 0:  # Long position
                # Exit if price reaches R4 (take profit) or breaks below S3 (stop)
                if price >= r4_6h[i] or price <= s3_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if price reaches S4 (take profit) or breaks above R3 (stop)
                if price <= s4_6h[i] or price >= r3_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Weekly trend filter: only long above weekly EMA, short below
            price_vs_weekly_ema = price - ema_1w_aligned[i]
            
            # Long entry: price at S3/S4 with bullish weekly trend (mean reversion or breakout)
            if price_vs_weekly_ema > 0:  # Bullish weekly trend
                if price <= s3_6h[i]:  # Mean reversion long at S3
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                elif price >= r4_6h[i]:  # Breakout long at R4
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
            else:  # Bearish weekly trend
                if price >= r3_6h[i]:  # Mean reversion short at R3
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                elif price <= s4_6h[i]:  # Breakout short at S4
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals