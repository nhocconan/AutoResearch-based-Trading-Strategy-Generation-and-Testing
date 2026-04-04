#!/usr/bin/env python3
"""
Experiment #5997: 4h Donchian(20) breakout + 1d/1w HTF bias + volume confirmation
HYPOTHESIS: Donchian breakouts on 4h aligned with 1d EMA50 trend and 1w volume regime filter capture sustained moves.
1d EMA50 provides medium-term trend bias, 1w volume ratio >1.2 confirms institutional participation.
ATR trailing stop manages risk. Target 75-200 trades over 4 years (19-50/year).
Works in both bull/bear: EMA50 filter avoids counter-trend entries in bear markets, volume confirmation avoids false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5997_4h_donchian20_1d_ema50_1w_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA50 trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    else:
        ema_1d_50_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 20:
        avg_volume_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
        volume_ratio_1w = df_1w['volume'].values / np.where(avg_volume_1w > 0, avg_volume_1w, 1)
        volume_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ratio_1w)
    else:
        volume_ratio_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (4h) ===
    avg_volume_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio_4h = volume / np.where(avg_volume_4h > 0, avg_volume_4h, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 50, 20) + 1  # Donchian, volume avg, ATR, 1d EMA50, 1w volume + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio_4h[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_50_aligned[i]) or np.isnan(volume_ratio_1w_aligned[i])):
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
        volume_confirmed_4h = volume_ratio_4h[i] > 1.5
        volume_confirmed_1w = volume_ratio_1w_aligned[i] > 1.2
        
        # HTF bias: 1d EMA50 trend
        above_ema = price > ema_1d_50_aligned[i]
        below_ema = price < ema_1d_50_aligned[i]
        
        # Entry conditions: 
        # Long: breakout up with volume (4h AND 1w) AND above 1d EMA50
        # Short: breakout down with volume (4h AND 1w) AND below 1d EMA50
        long_setup = breakout_up and volume_confirmed_4h and volume_confirmed_1w and above_ema
        short_setup = breakout_down and volume_confirmed_4h and volume_confirmed_1w and below_ema
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</p>