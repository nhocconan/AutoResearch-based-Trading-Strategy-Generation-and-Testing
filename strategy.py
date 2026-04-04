#!/usr/bin/env python3
"""
Experiment #5039: 6h Donchian(20) Breakout + 12h HTF Trend Filter + Volume Spike + ATR Stoploss
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with 12h EMA trend (EMA20/EMA50) capture strong momentum with lower frequency. The 12h EMA acts as a trend filter: only take longs when price > EMA20 > EMA50 (bullish alignment) and shorts when price < EMA20 < EMA50 (bearish alignment). Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5039_6h_donchian20_12h_ema_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: EMA20 and EMA50 for trend alignment ===
    if len(df_12h) >= 50:  # Need sufficient data for EMA50
        close_12h = df_12h['close'].values
        ema20_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        
        # Trend conditions: bullish = price > EMA20 > EMA50, bearish = price < EMA20 < EMA50
        ema20_gt_ema50 = ema20_12h > ema50_12h
        ema20_lt_ema50 = ema20_12h < ema50_12h
        
        # Align to 6h timeframe
        ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
        ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
        ema20_gt_ema50_aligned = align_htf_to_ltf(prices, df_12h, ema20_gt_ema50.astype(np.float64))
        ema20_lt_ema50_aligned = align_htf_to_ltf(prices, df_12h, ema20_lt_ema50.astype(np.float64))
    else:
        ema20_12h_aligned = np.full(n, np.nan)
        ema50_12h_aligned = np.full(n, np.nan)
        ema20_gt_ema50_aligned = np.full(n, np.nan)
        ema20_lt_ema50_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, Volume MA, ATR, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(ema20_gt_ema50_aligned[i]) or np.isnan(ema20_lt_ema50_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend alignment from 12h
        bullish_trend = bool(ema20_gt_ema50_aligned[i] == 1.0)
        bearish_trend = bool(ema20_lt_ema50_aligned[i] == 1.0)
        
        # Donchian breakout conditions with 12h EMA trend filter
        # Long: Donchian breakout above high_roll AND bullish trend alignment AND volume
        # Short: Donchian breakdown below low_roll AND bearish trend alignment AND volume
        breakout_long = (price >= high_roll[i]) and bullish_trend and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_trend and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals