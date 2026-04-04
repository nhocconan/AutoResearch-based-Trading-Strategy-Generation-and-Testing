#!/usr/bin/env python3
"""
Experiment #4358: 1d Donchian(20) Breakout + Weekly Volume Spike + Chop Regime Filter
HYPOTHESIS: Donchian(20) breakouts on daily timeframe aligned with weekly volume confirmation (>2.0x average) and choppy market filter (Choppiness Index > 61.8 = range) capture institutional accumulation/distribution in ranging markets. Weekly volume confirms participation, chop filter ensures mean-reversion context. Works in bull via upward breakouts in accumulation ranges, in bear via downward breakouts in distribution ranges. Targets 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4358_1d_donchian20_1w_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1w OHLC for volume and chop ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        weekly_volume = df_1w['volume'].values
        
        # Weekly volume MA(4) for confirmation
        weekly_vol_ma = pd.Series(weekly_volume).rolling(window=4, min_periods=4).mean().values
        
        # Weekly True Range for Choppiness Index
        tr1 = weekly_high[1:] - weekly_low[1:]
        tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
        tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
        weekly_tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        weekly_atr = pd.Series(weekly_tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Weekly Chop: (sum(ATR14) / (max(high,n) - min(low,n))) * 100
        weekly_sum_atr14 = pd.Series(weekly_atr).rolling(window=14, min_periods=14).sum().values
        weekly_maxh = pd.Series(weekly_high).rolling(window=14, min_periods=14).max().values
        weekly_minl = pd.Series(weekly_low).rolling(window=14, min_periods=14).min().values
        weekly_chop = np.where((weekly_maxh - weekly_minl) != 0,
                               (weekly_sum_atr14 / (weekly_maxh - weekly_minl)) * 100, 100)
        
        # Align HTF arrays to LTF
        weekly_vol_ma_aligned = align_htf_to_ltf(prices, df_1w, weekly_vol_ma)
        weekly_chop_aligned = align_htf_to_ltf(prices, df_1w, weekly_chop)
    else:
        weekly_vol_ma_aligned = np.full(n, np.nan)
        weekly_chop_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 14, 4, 14)  # Donchian, ATR, vol MA, chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_vol_ma_aligned[i]) or np.isnan(weekly_chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
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
        # Require weekly volume confirmation (> 2.0x average) to filter noise
        volume_confirm = weekly_volume[i] > 2.0 * weekly_vol_ma_aligned[i] if not np.isnan(weekly_vol_ma_aligned[i]) else False
        
        # Chop regime filter: CHOP > 61.8 = ranging market (mean revert context)
        chop_regime = weekly_chop_aligned[i] > 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + volume + chop regime
        long_entry = breakout_up and volume_confirm and chop_regime
        
        # Short conditions: downward breakout + volume + chop regime
        short_entry = breakout_down and volume_confirm and chop_regime
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals