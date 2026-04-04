#!/usr/bin/env python3
"""
Experiment #4624: 1d Donchian(20) Breakout + Weekly Trend + Volume Confirmation
HYPOTHESIS: Daily price breaking 20-day Donchian channels with weekly EMA(21) trend alignment and volume confirmation (>1.3x average) captures strong momentum moves. Uses discrete sizing (0.25) and ATR(14) trailing stop (2.0x) for risk management. Target: 15-30 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4624_1d_donchian20_weekly_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for weekly EMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(21) for trend filter
    if len(df_1w) >= 1:
        weekly_close = df_1w['close'].values
        weekly_ema = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    else:
        weekly_ema = np.array([])
    
    # Align weekly EMA to daily timeframe
    if len(weekly_ema) > 0:
        weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    else:
        weekly_ema_aligned = np.full(n, np.nan)
    
    # === Daily Indicators: Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Daily Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Daily Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 21, 20)  # Donchian, weekly EMA, volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.3x average volume)
        vol_confirm = vol_ratio[i] > 1.3
        
        # Trend filter: price above/below weekly EMA
        price_above_weekly = price > weekly_ema_aligned[i]
        price_below_weekly = price < weekly_ema_aligned[i]
        
        # Breakout conditions: price breaks Donchian channels with volume and trend confirmation
        breakout_long = price > donchian_upper[i] and vol_confirm and price_above_weekly
        breakout_short = price < donchian_lower[i] and vol_confirm and price_below_weekly
        
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