#!/usr/bin/env python3
"""
Experiment #2307: 6h Camarilla Pivot + 1d/1w Trend Filter + Volume Spike
HYPOTHESIS: Camarilla pivot levels on 6h act as intraday support/resistance with higher timeframe trend bias.
- Primary: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) calculated from prior 1d OHLC
- HTF: 1d EMA(50) and 1w EMA(50) trend alignment (both must agree for bias)
- Entry: Long at S3 with 1d/1w uptrend + volume spike; Short at R3 with 1d/1w downtrend + volume spike
- Exit: Opposite pivot level (S4 for longs, R4 for shorts) or ATR(14) stop (2*ATR)
- Volume: Require > 2.0x 20-bar average spike to confirm participation
- Target: 75-150 total trades over 4 years (19-37/year) - suitable for 6h timeframe
- Works in bull markets (breakouts at R4/S4) and bear markets (mean reversion at R3/S3)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2307_6h_camarilla_1d1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend and prior OHLC (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for EMA trend ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Prior 1d OHLC for Camarilla, ATR(14), Volume MA(20) ===
    # Get prior 1d OHLC for each 6h bar (shifted by 1 to avoid look-ahead)
    # Since 1d data is aligned to 6h, we can use the previous 1d bar's OHLC
    # We'll shift the 1d OHLC arrays by 1 to get prior day's values
    if len(high_1d) > 0:
        high_1d_prev = np.roll(high_1d, 1)
        low_1d_prev = np.roll(low_1d, 1)
        close_1d_prev = np.roll(close_1d, 1)
        high_1d_prev[0] = np.nan  # First bar has no prior day
        low_1d_prev[0] = np.nan
        close_1d_prev[0] = np.nan
    else:
        high_1d_prev = np.array([])
        low_1d_prev = np.array([])
        close_1d_prev = np.array([])
    
    # Align prior 1d OHLC to 6h timeframe
    high_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, high_1d_prev)
    low_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, low_1d_prev)
    close_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, close_1d_prev)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    pivot = (high_1d_prev_aligned + low_1d_prev_aligned + close_1d_prev_aligned) / 3.0
    rng = high_1d_prev_aligned - low_1d_prev_aligned
    camarilla_r4 = close_1d_prev_aligned + rng * 1.1 / 2.0
    camarilla_r3 = close_1d_prev_aligned + rng * 1.1 / 4.0
    camarilla_s3 = close_1d_prev_aligned - rng * 1.1 / 4.0
    camarilla_s4 = close_1d_prev_aligned - rng * 1.1 / 2.0
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches S4 (strong support break)
                elif price <= camarilla_s4[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches R4 (strong resistance break)
                elif price >= camarilla_r4[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 1d and 1w trend alignment for bias filter
        trend_bias_1d = trend_1d_aligned[i]
        trend_bias_1w = trend_1w_aligned[i]
        
        # Only trade when both timeframes agree
        if trend_bias_1d == trend_bias_1w:
            trend_bias = trend_bias_1d  # Either 1 or -1
        else:
            trend_bias = 0  # No clear trend, skip
        
        # Volume confirmation: require volume spike (> 2.0x average - strict to limit trades)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Long entry: price at S3 with uptrend on both timeframes
            if trend_bias > 0 and abs(price - camarilla_s3[i]) < 0.001 * camarilla_s3[i]:  # Near S3
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price at R3 with downtrend on both timeframes
            elif trend_bias < 0 and abs(price - camarilla_r3[i]) < 0.001 * camarilla_r3[i]:  # Near R3
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals