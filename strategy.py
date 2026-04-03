#!/usr/bin/env python3
"""
Experiment #415: 6h Weekly Pivot + Volume Spike + 1d Trend Filter

HYPOTHESIS: Weekly pivot points (from 1w data) provide key institutional support/resistance levels. 
Combined with 6h price action at these levels, 12h volume spike confirmation (>2.0x average), 
and 1d trend filter (price > EMA50 for longs, < EMA50 for shorts), this strategy captures 
high-probability mean reversions and breakouts. Weekly pivots are less noisy than daily and 
align with smart money cycles. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
Works in bull markets (breakout continuation) and bear markets (mean reversion at pivot levels).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points for each 6h bar using prior week's OHLC
    weekly_pivot = np.full(n, np.nan)
    weekly_r1 = np.full(n, np.nan)
    weekly_s1 = np.full(n, np.nan)
    weekly_r2 = np.full(n, np.nan)
    weekly_s2 = np.full(n, np.nan)
    weekly_r3 = np.full(n, np.nan)
    weekly_s3 = np.full(n, np.nan)
    
    # For each 6h bar, get the prior completed 1w bar's OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 1w bar before current 6h bar
        prior_1w_bars = df_1w[df_1w['open_time'] < current_time]
        if len(prior_1w_bars) > 0:
            prev_week = prior_1w_bars.iloc[-1]
            ph = prev_week['high']
            pl = prev_week['low']
            pc = prev_week['close']
            
            # Standard pivot point formulas
            pivot = (ph + pl + pc) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - pl
            weekly_s1[i] = 2 * pivot - ph
            weekly_r2[i] = pivot + (ph - pl)
            weekly_s2[i] = pivot - (ph - pl)
            weekly_r3[i] = ph + 2 * (pivot - pl)
            weekly_s3[i] = pl - 2 * (ph - pivot)
        else:
            # Not enough prior data
            weekly_pivot[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
            weekly_r2[i] = np.nan
            weekly_s2[i] = np.nan
            weekly_r3[i] = np.nan
            weekly_s3[i] = np.nan
    
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
        if (np.isnan(weekly_pivot[i]) or np.isnan(weekly_r1[i]) or np.isnan(weekly_s1[i]) or
            np.isnan(weekly_r2[i]) or np.isnan(weekly_s2[i]) or np.isnan(weekly_r3[i]) or 
            np.isnan(weekly_s3[i]) or np.isnan(vol_ratio_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (price > 1d EMA50 for long, < for short) ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
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
                # Take profit at weekly S3 (strong support) or R3 (strong resistance)
                if close[i] >= weekly_r3[i] or close[i] <= weekly_s3[i]:
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
                # Take profit at weekly R3 (strong resistance) or S3 (strong support)
                if close[i] >= weekly_r3[i] or close[i] <= weekly_s3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at S1 (mean reversion) OR break above R1 with volume
        long_condition = (
            (abs(close[i] - weekly_s1[i]) / weekly_s1[i] < 0.002 and price_above_1d_ema) or  # S1 mean reversion in uptrend
            (close[i] > weekly_r1[i] and volume_spike and price_above_1d_ema)  # Breakout with volume
        )
        
        # Short: Price at R1 (mean reversion) OR break below S1 with volume
        short_condition = (
            (abs(close[i] - weekly_r1[i]) / weekly_r1[i] < 0.002 and price_below_1d_ema) or  # R1 mean reversion in downtrend
            (close[i] < weekly_s1[i] and volume_spike and price_below_1d_ema)  # Breakdown with volume
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