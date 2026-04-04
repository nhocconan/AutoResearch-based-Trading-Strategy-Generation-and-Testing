#!/usr/bin/env python3
"""
Experiment #5277: 4h Donchian(20) breakout + volume confirmation + 1d EMA trend filter
HYPOTHESIS: On 4h timeframe, price breaking above/below 20-period Donchian channel with above-average volume, filtered by 1d EMA50 trend direction, captures strong momentum moves while avoiding false breakouts in ranging markets. The Donchian breakout provides clear entry/exit levels, volume confirmation ensures institutional participation, and the 1d EMA50 filter aligns with higher timeframe trend to reduce whipsaws. Works in bull markets by buying breakouts above channel and in bear markets by selling breakdowns below channel. Uses discrete position sizing (0.25) to balance profit potential with drawdown control. Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5277_4h_donchian20_vol_1d_ema_v1"
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
    
    # === HTF: 1d data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper band: 20-period high
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 50)  # Donchian, volume MA, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (optional) ---
        hour = hours[i]
        # 4h timeframe: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        # Avoid 00:00-04:00 UTC if needed (typically lower volume)
        # Uncomment below to filter: only trade 04:00-20:00 UTC
        # if hour < 4 or hour >= 20:
        #     signals[i] = 0.0
        #     continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # --- Exit Logic: Close position when price reverses back into Donchian channel ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit when price closes below Donchian lower band
                if price < donch_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit when price closes above Donchian upper band
                if price > donch_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Breakout conditions
        breakout_up = price > donch_high[i-1]  # Price breaks above previous upper band
        breakout_down = price < donch_low[i-1]  # Price breaks below previous lower band
        
        # Volume confirmation: current volume > 20-period average
        volume_confirm = vol > vol_ma[i]
        
        # Trend filter from 1d EMA50
        trend_bullish = price > ema_50_aligned[i]
        trend_bearish = price < ema_50_aligned[i]
        
        # Entry conditions: Breakout + volume confirmation + trend alignment
        if breakout_up and volume_confirm and trend_bullish:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirm and trend_bearish:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals