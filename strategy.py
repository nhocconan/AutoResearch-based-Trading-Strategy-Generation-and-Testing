#!/usr/bin/env python3
"""
Experiment #052: 12h Camarilla Pivot + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 12h timeframe,
combined with 1d volume spike confirmation and 1w trend filter (price > EMA50), creates a
robust strategy that works in both bull and bear markets. The 12h timeframe targets 12-37
trades/year (50-150 total over 4 years) to minimize fee drag. Camarilla levels provide
mathematically derived support/resistance, volume confirms institutional participation,
and the weekly trend filter ensures alignment with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_trend_v1"
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
    
    # === 12h Indicators ===
    # Calculate Camarilla pivot levels for each 12h bar using previous day's OHLC
    # We need to map each 12h bar to the prior 1d bar's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_r3_s4 = np.full(n, np.nan)  # Midpoint between R3 and S4 for breakout
    
    # For each 12h bar, get the prior 1d bar's OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 1d bar before current 12h bar
        prior_1d_bars = df_1d[df_1d['open_time'] < current_time]
        if len(prior_1d_bars) > 0:
            prev_day = prior_1d_bars.iloc[-1]
            ph = prev_day['high']
            pl = prev_day['low']
            pc = prev_day['close']
            
            # Camarilla formulas
            range_ = ph - pl
            camarilla_r3[i] = pc + range_ * 1.1 / 4
            camarilla_s3[i] = pc - range_ * 1.1 / 4
            camarilla_r4[i] = pc + range_ * 1.1 / 2
            camarilla_s4[i] = pc - range_ * 1.1 / 2
            camarilla_r3_s4[i] = (camarilla_r3[i] + camarilla_s4[i]) / 2
        else:
            # Not enough prior data
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_r4[i] = np.nan
            camarilla_s4[i] = np.nan
            camarilla_r3_s4[i] = np.nan
    
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
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (price > 1w EMA50 for long, < for short) ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
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
                if close[i] >= camarilla_r4[i] or close[i] <= camarilla_s4[i]:
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
                if close[i] >= camarilla_r4[i] or close[i] <= camarilla_s4[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at S3 (mean reversion) OR break above R3_S4 midpoint with volume
        long_condition = (
            (close[i] <= camarilla_s3[i] * 1.001 and price_above_1w_ema) or  # S3 mean reversion in uptrend
            (close[i] > camarilla_r3_s4[i] and volume_spike and price_above_1w_ema)  # Breakout with volume
        )
        
        # Short: Price at R3 (mean reversion) OR break below R3_S4 midpoint with volume
        short_condition = (
            (close[i] >= camarilla_r3[i] * 0.999 and price_below_1w_ema) or  # R3 mean reversion in downtrend
            (close[i] < camarilla_r3_s4[i] and volume_spike and price_below_1w_ema)  # Breakdown with volume
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