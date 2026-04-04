#!/usr/bin/env python3
"""
Experiment #5075: 6h Donchian(20) Breakout + 1w/1d HTF Regime + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly (1w) and daily (1d) HTF regime filters capture strong momentum with lower frequency. Weekly trend (price > weekly EMA200) filters for bull/bear regime, while daily pivot levels (R3/S3/R4/S4) provide institutional support/resistance. Volume > 2x average confirms participation. Designed for 12-37 trades/year on 6h to minimize fee drag. Works in bull (breakouts through R4 in uptrend) and bear (breakdowns through S4 in downtrend) by using HTF trend as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5075_6h_donchian20_1w_1d_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w and 1d data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # === 1w Indicators: Weekly EMA200 for trend regime ===
    if len(df_1w) >= 200:
        weekly_ema200 = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
        weekly_ema200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema200)
        weekly_trend_up = close >= weekly_ema200_aligned  # Price above weekly EMA200 = bull regime
        weekly_trend_down = close < weekly_ema200_aligned  # Price below weekly EMA200 = bear regime
    else:
        weekly_ema200_aligned = np.full(n, np.nan)
        weekly_trend_up = np.zeros(n, dtype=bool)
        weekly_trend_down = np.zeros(n, dtype=bool)
    
    # === 1d Indicators: Daily Pivot Points (using prior day's OHLC) ===
    if len(df_1d) >= 2:
        # Prior day's OHLC
        prior_high = df_1d['high'].shift(1).values  # Shifted for completed day only
        prior_low = df_1d['low'].shift(1).values
        prior_close = df_1d['close'].shift(1).values
        
        # Daily Pivot Point = (Prior Day H + L + C) / 3
        pp = (prior_high + prior_low + prior_close) / 3.0
        
        # Daily Support/Resistance Levels
        rng = prior_high - prior_low
        r1 = (2 * pp) - prior_low
        s1 = (2 * pp) - prior_high
        r2 = pp + rng
        s2 = pp - rng
        r3 = prior_high + 2 * (pp - prior_low)
        s3 = prior_low - 2 * (prior_high - pp)
        r4 = pp + 3 * rng
        s4 = pp - 3 * rng
        
        # Align to 6h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (2x spike) ===
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
    
    warmup = max(20, 20, 14, 200)  # Donchian, Volume MA, ATR, Weekly EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_ema200_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Regime filters from HTF
        is_bull_regime = weekly_trend_up[i]
        is_bear_regime = weekly_trend_down[i]
        
        # Donchian breakout conditions with HTF regime and pivot alignment
        # Long: In bull regime, breakout above R4 OR above R3 with volume (if failing mean reversion)
        # Short: In bear regime, breakdown below S4 OR below S3 with volume (if failing mean reversion)
        breakout_long = is_bull_regime and vol_confirm and (
            (price >= r4_aligned[i]) or  # Strong breakout through daily R4
            ((price >= r3_aligned[i]) and (price <= high_roll[i]))  # Break above R3 but below Donchian high (false breakout fade)
        )
        
        breakout_short = is_bear_regime and vol_confirm and (
            (price <= s4_aligned[i]) or  # Strong breakdown through daily S4
            ((price <= s3_aligned[i]) and (price >= low_roll[i]))  # Break below S3 but above Donchian low (false breakdown fade)
        )
        
        # Alternative: Simple Donchian breakout with volume in respective regime
        # Long: Donchian breakout above high_roll with volume in bull regime
        # Short: Donchian breakdown below low_roll with volume in bear regime
        donchian_long = is_bull_regime and vol_confirm and (price >= high_roll[i])
        donchian_short = is_bear_regime and vol_confirm and (price <= low_roll[i])
        
        # Final entry conditions (using Donchian breakout as primary signal)
        if donchian_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif donchian_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals