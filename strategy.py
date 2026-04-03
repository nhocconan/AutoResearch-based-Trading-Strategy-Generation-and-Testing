#!/usr/bin/env python3
"""
Experiment #394: 1h Camarilla Pivot + 4h/1d Volume Spike + Trend Filter

HYPOTHESIS: Camarilla pivot levels (S3/R3 for mean reversion, R3/S4 breakout) on 1h timeframe,
combined with 4h volume spike confirmation and 1d trend filter (price > EMA50), creates a 
robust strategy for both bull and bear markets. Camarilla provides mathematically derived 
S/R levels, volume confirms institutional participation, and 1d trend filter ensures 
alignment with higher timeframe direction. Uses session filter (08-20 UTC) to reduce noise.
Target: 60-150 total trades over 4 years = 15-37/year on 1h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for volume spike (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate volume ratio (current vs 20-period average) on 4h
    if len(df_4h) >= 20:
        vol_4h = df_4h['volume'].values
        vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_4h = np.zeros(len(vol_4h))
        vol_ratio_4h[20:] = vol_4h[20:] / vol_ma_20[20:]
        vol_ratio_4h[:20] = 1.0  # Neutral for warmup
        vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    else:
        vol_ratio_4h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Camarilla pivot levels ===
    # Calculate Camarilla pivot levels for each 1h bar using prior 1d bar's OHLC
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r3_s4 = np.full(n, np.nan)  # Midpoint between R3 and S4 for breakout
    
    # For each 1h bar, get the prior 1d bar's OHLC
    for i in range(n):
        current_time = open_time[i]
        # Find the most recent completed 1d bar before current 1h bar
        prior_1d_bars = df_1d[df_1d['open_time'] < current_time]
        if len(prior_1d_bars) > 0:
            prev_day = prior_1d_bars.iloc[-1]
            ph = prev_day['high']
            pl = prev_day['low']
            pc = prev_day['close']
            
            # Camarilla formulas
            range_ = ph - pl
            camarilla_s3[i] = pc - range_ * 1.1 / 4
            camarilla_r3[i] = pc + range_ * 1.1 / 4
            camarilla_r4 = pc + range_ * 1.1 / 2  # R4
            camarilla_s4 = pc - range_ * 1.1 / 2  # S4
            camarilla_r3_s4[i] = (camarilla_r3[i] + camarilla_s4) / 2
        else:
            # Not enough prior data
            camarilla_s3[i] = np.nan
            camarilla_r3[i] = np.nan
            camarilla_r3_s4[i] = np.nan
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_s3[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_r3_s4[i]) or np.isnan(vol_ratio_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (price > 1d EMA50 for long, < for short) ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_4h_aligned[i] > 1.5
        
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
                # Take profit at Camarilla S4 (strong support) or R4 (strong resistance)
                if close[i] >= camarilla_r3[i] + (camarilla_r3[i] - camarilla_s3[i]) or \
                   close[i] <= camarilla_s3[i] - (camarilla_r3[i] - camarilla_s3[i]):
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
                # Take profit at Camarilla R4 (strong resistance) or S4 (strong support)
                if close[i] >= camarilla_r3[i] + (camarilla_r3[i] - camarilla_s3[i]) or \
                   close[i] <= camarilla_s3[i] - (camarilla_r3[i] - camarilla_s3[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at S3 (mean reversion) OR break above R3 with volume
        long_condition = (
            (close[i] <= camarilla_s3[i] * 1.001 and price_above_1d_ema) or  # S3 mean reversion in uptrend
            (close[i] > camarilla_r3[i] and volume_spike and price_above_1d_ema)  # Breakout with volume
        )
        
        # Short: Price at R3 (mean reversion) OR break below S3 with volume
        short_condition = (
            (close[i] >= camarilla_r3[i] * 0.999 and price_below_1d_ema) or  # R3 mean reversion in downtrend
            (close[i] < camarilla_s3[i] and volume_spike and price_below_1d_ema)  # Breakdown with volume
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