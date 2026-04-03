#!/usr/bin/env python3
"""
Experiment #474: 1h strategy with 4h/1d HTF filters
HYPOTHESIS: Use 4h Donchian(20) for trend direction and 1d EMA(50) for regime filter, with 1h RSI(14) pullback entries. This combines HTF structure with lower TF timing to target 15-37 trades/year. Session filter (08-20 UTC) reduces noise. Discrete sizing at 0.20 controls risk and fee drag. Designed to work in both bull (breakouts with trend) and bear (mean reversion in range) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_474_1h_donchian4h_ema1d_rsi_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for Donchian(20) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    highest_high_4h = high_4h.rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_4h = low_4h.rolling(window=20, min_periods=20).min().shift(1).values
    donchian_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    donchian_l_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # === HTF: 1d data for EMA(50) regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: RSI(14) for pullback entries ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral during warmup
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # sufficient for 1d EMA(50) + 1h indicators
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_4h_aligned[i]) or np.isnan(donchian_l_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: 1d EMA(50) ---
        price_above_ema = price > ema_1d_aligned[i]
        price_below_ema = price < ema_1d_aligned[i]
        
        # --- HTF Trend: 4h Donchian(20) Breakout Direction ---
        donchian_breakout_up = price > donchian_4h_aligned[i]
        donchian_breakout_down = price < donchian_l_4h_aligned[i]
        
        # --- 1h RSI(14) Pullback Conditions ---
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 50  # pullback from oversold
        rsi_pullback_short = rsi[i] < 60 and rsi[i] > 50  # pullback from overbought
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 24 bars (~1 day) to avoid overtrading
            if bars_since_entry > 24:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: price above 1d EMA (bull regime) + 4h Donchian breakout up + RSI pullback from oversold
        if price_above_ema and donchian_breakout_up and rsi_pullback_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: price below 1d EMA (bear regime) + 4h Donchian breakout down + RSI pullback from overbought
        elif price_below_ema and donchian_breakout_down and rsi_pullback_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals