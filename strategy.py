#!/usr/bin/env python3
"""
Experiment #6158: 1d Donchian(20) breakout + 1w CAMARILLA pivot + volume confirmation
HYPOTHESIS: Daily Donchian breakouts aligned with weekly CAMARILLA pivot levels (H3/L3) capture structural moves.
Weekly pivot from 1w provides longer-term bias: price above weekly H3 = bullish, below weekly L3 = bearish.
Volume >1.5x average confirms strong participation. ATR trailing stop manages risk.
Discrete sizing (0.25) minimizes fee churn. Target: 30-100 trades over 4 years.
Timeframe: 1d. HTF: 1w for CAMARILLA pivot calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6158_1d_donchian20_1w_camarilla_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for CAMARILLA pivot (using prior week's high/low/close) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly CAMARILLA pivot levels from prior week's OHLC
        # H3 = close + 1.1*(high - low)
        # L3 = close - 1.1*(high - low)
        # Use prior week's data (shift by 1 to avoid look-ahead)
        weekly_high = pd.Series(df_1w['high'].values).shift(1).values
        weekly_low = pd.Series(df_1w['low'].values).shift(1).values
        weekly_close = pd.Series(df_1w['close'].values).shift(1).values
        weekly_range = weekly_high - weekly_low
        camarilla_h3 = weekly_close + 1.1 * weekly_range
        camarilla_l3 = weekly_close - 1.1 * weekly_range
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 1d Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 1) + 1  # Donchian, volume avg, ATR, weekly pivot shift + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter for stronger signals
        
        # Multi-timeframe trend filter: price relative to 1w CAMARILLA levels
        bullish_bias = price > camarilla_h3_aligned[i]  # Above weekly H3 = bullish
        bearish_bias = price < camarilla_l3_aligned[i]  # Below weekly L3 = bearish
        
        # Entry conditions:
        # Long: breakout up with volume AND bullish bias above weekly H3
        # Short: breakout down with volume AND bearish bias below weekly L3
        long_entry = breakout_up and volume_confirmed and bullish_bias
        short_entry = breakout_down and volume_confirmed and bearish_bias
        
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