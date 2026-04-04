#!/usr/bin/env python3
"""
Experiment #5272: 12h Donchian(20) Breakout + 1d EMA Trend + Volume Spike
HYPOTHESIS: On 12h timeframe, price breaking above/below the 20-period Donchian channel (highest high/lowest low) captures strong momentum moves. The 1d EMA50 acts as a regime filter (bullish when price > EMA50, bearish when price < EMA50) to avoid counter-trend trades. Volume confirmation (current volume > 1.5x 20-period average) ensures breakouts have participation. Designed for 12-25 trades/year on 12h timeframe (50-100 total over 4 years) to minimize fee drag. Works in bull markets by catching uptrend continuations and in bear markets by catching downtrend continuations, while avoiding false breakouts in low-volume or choppy conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5272_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # === HTF: 1d data for regime filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Upper band: highest high over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: lowest low over 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume Spike (current volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 50)  # Donchian, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (12h timeframe, less restrictive) ---
        # 12h candles already filter to specific sessions, so we can use full day
        # Optional: avoid low liquidity periods if needed
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when price crosses opposite Donchian band or regime changes ---
        if in_position:
            # Check regime consistency
            regime_bullish = price > ema_50_aligned[i]
            regime_bearish = price < ema_50_aligned[i]
            
            # Exit conditions:
            # 1. Price crosses opposite Donchian band (mean reversion signal)
            # 2. Regime changes (price crosses 1d EMA50)
            if position_side > 0:  # Long position
                if (price < donchian_low[i]) or (not regime_bullish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if (price > donchian_high[i]) or (not regime_bearish):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout
        breakout_up = price > donchian_high[i]
        breakout_down = price < donchian_low[i]
        
        # Regime filter from 1d
        regime_bullish = price > ema_50_aligned[i]
        regime_bearish = price < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry conditions: Donchian breakout + regime match + volume confirmation
        if breakout_up and regime_bullish and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and regime_bearish and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals