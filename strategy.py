#!/usr/bin/env python3
"""
Experiment #5276: 12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike
HYPOTHESIS: On 12h timeframe, Donchian channel breakouts capture strong momentum moves. 
Filtering by 1d EMA50 ensures we trade with the higher timeframe trend, reducing whipsaws. 
Volume confirmation (current volume > 1.5 * 20-period average) ensures breakouts have conviction. 
ATR-based stoploss (signal → 0 when price moves 2*ATR against position) manages risk. 
Designed for 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to minimize fee drag. 
Works in bull markets by catching breakouts above upper channel and in bear markets by catching 
breakdowns below lower channel, while avoiding false breakouts in low-volume or choppy conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5276_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    # Upper channel: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: ATR (14-period) for stoploss ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume Spike Filter ===
    # 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    # Volume spike: current volume > 1.5 * 20-period average
    volume_spike = volume > (1.5 * vol_ma)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 50, 14, 20)  # Donchian, EMA50, ATR, volume MA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (optional) ---
        hour = hours[i]
        # Trade all hours for 12h timeframe (less restrictive)
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_spike[i] if not np.isnan(volume_spike[i]) else False
        
        # --- Exit Logic: Close position on reversal or stoploss ---
        if in_position:
            # Stoploss: 2 * ATR against position
            if position_side > 0:  # Long position
                stop_price = entry_price - 2.0 * atr[i]
                if price < stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian reversal (price closes below lower channel)
                if price < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on trend change (price crosses 1d EMA50)
                if price < ema_50_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                signals[i] = SIZE
            else:  # Short position
                stop_price = entry_price + 2.0 * atr[i]
                if price > stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian reversal (price closes above upper channel)
                if price > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on trend change (price crosses 1d EMA50)
                if price > ema_50_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long entry: price breaks above upper Donchian channel + uptrend + volume spike
        long_entry = (price > donchian_high[i]) and (price > ema_50_aligned[i]) and vol_ok
        # Short entry: price breaks below lower Donchian channel + downtrend + volume spike
        short_entry = (price < donchian_low[i]) and (price < ema_50_aligned[i]) and vol_ok
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals