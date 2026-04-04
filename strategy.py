#!/usr/bin/env python3
"""
Experiment #5273: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: On 4h timeframe, price breaking above/below 20-period Donchian channel with 12h HMA trend confirmation and volume spike captures strong momentum moves. Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. Designed for 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag. Works in bull markets by catching upside breakouts and in bear markets by catching downside breakdowns, while avoiding ranging conditions via volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5273_4h_donchian20_12h_hma_vol_v1"
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
    
    # === HTF: 12h data for HMA trend filter ===
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # === 4h Indicators: ATR for stoploss (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (optional) ---
        hour = hours[i]
        # Trade during major market hours: 00-06 UTC (Asia), 07-11 UTC (Europe/London open), 12-16 UTC (US open)
        # Avoid 17-23 UTC (low liquidity between sessions)
        if 17 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on Donchian reversal or ATR stoploss ---
        if in_position:
            # Check for Donchian reversal (price crosses opposite channel)
            if position_side > 0:  # Long position
                # Exit: price breaks below Donchian low OR ATR stoploss hit
                if price < donchian_low[i] or price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit: price breaks above Donchian high OR ATR stoploss hit
                if price > donchian_high[i] or price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout
        breakout_long = price > donchian_high[i]
        breakout_short = price < donchian_low[i]
        
        # 12h HMA trend filter
        hma_bullish = price > hma_12h_aligned[i]
        hma_bearish = price < hma_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry conditions: Donchian breakout + HMA trend + volume confirmation
        if breakout_long and hma_bullish and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_short and hma_bearish and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    values = pd.Series(values)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = values.ewm(span=half_period, adjust=False).mean()
    wma_full = values.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_period, adjust=False).mean()
    
    return hma.values