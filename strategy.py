#!/usr/bin/env python3
"""
Experiment #5389: 4h Donchian(20) breakout + 1d/1w HTF trend + volume confirmation
HYPOTHESIS: On 4h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 2.0x average and aligned with both 1d EMA(50) and 1w EMA(200) trend 
captures strong momentum moves while minimizing false breakouts. The dual timeframe 
trend filter (1d + 1w) ensures we only trade in the direction of both intermediate 
and long-term trends, reducing whipsaws in ranging markets. Discrete position sizing 
(0.25) and ATR-based trailing stoploss (2.5x ATR) control risk. Target: 19-50 trades/year 
(75-200 total over 4 years) to minimize fee drag while maintaining statistical significance. 
Works in bull markets via breakouts above rising dual-timeframe EMAs and in bear markets 
via short breakdowns below falling dual-timeframe EMAs.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5389_4h_donchian20_1d_1w_ema_vol_v1"
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
    
    # === HTF: 1d data for EMA(50) trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        close_1d = pd.Series(df_1d['close'].values)
        ema_1d_50 = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for EMA(200) trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 200:
        close_1w = pd.Series(df_1w['close'].values)
        ema_1w_200 = close_1w.ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50, 200)  # Donchian, volume avg, ATR, 1d EMA, 1w EMA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below highest since entry
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price crosses below either EMA (trend reversal)
                if price <= stop_price or price <= donchian_low[i] or price < ema_1d_aligned[i] or price < ema_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.5 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price crosses above either EMA (trend reversal)
                if price >= stop_price or price >= donchian_high[i] or price > ema_1d_aligned[i] or price > ema_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Dual timeframe trend filter
        # Long: price above both 1d EMA(50) and 1w EMA(200) (bullish bias)
        # Short: price below both 1d EMA(50) and 1w EMA(200) (bearish bias)
        trend_bias_up = price > ema_1d_aligned[i-1] and price > ema_1w_aligned[i-1]
        trend_bias_down = price < ema_1d_aligned[i-1] and price < ema_1w_aligned[i-1]
        
        # Entry conditions
        if breakout_up and volume_confirmed and trend_bias_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and trend_bias_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>