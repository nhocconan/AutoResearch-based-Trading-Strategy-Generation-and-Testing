#!/usr/bin/env python3
"""
Experiment #5988: 12h Donchian(20) breakout + 1w/1d HTF bias + volume confirmation
HYPOTHESIS: Donchian breakouts on 12h timeframe aligned with weekly trend (price > weekly EMA50) 
and daily momentum (daily close > daily open) capture sustained moves with lower noise. 
Weekly EMA50 provides structural bias resilient to 12h noise, daily candle direction confirms 
short-term momentum. Volume >1.5x average confirms breakout strength. ATR trailing stop 
manages risk. Target 50-150 trades over 4 years (12-37/year). Works in both bull/bear: 
weekly trend filter prevents counter-trend entries in bear markets, daily momentum avoids 
false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5988_12h_donchian20_1w1d_bias_vol_v1"
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
    
    # === HTF: 1d data for daily candle direction ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Daily bullish bias: close > open
        daily_bullish = (df_1d['close'] > df_1d['open']).astype(float).values
        daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    else:
        daily_bullish_aligned = np.zeros(n)
    
    # === HTF: 1w data for weekly trend (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        weekly_ema50 = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    else:
        weekly_ema50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 12h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 50) + 1  # Donchian, volume avg, ATR, weekly EMA + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_ema50_aligned[i])):
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
        volume_confirmed = volume_ratio[i] > 1.5
        
        # HTF bias: 
        # Weekly trend: price above/below weekly EMA50
        above_weekly_trend = price > weekly_ema50_aligned[i]
        below_weekly_trend = price < weekly_ema50_aligned[i]
        
        # Daily momentum: bullish/bearish daily candle
        daily_bull = daily_bullish_aligned[i] > 0.5
        daily_bear = daily_bullish_aligned[i] < 0.5
        
        # Entry conditions: 
        # Long: breakout up with volume AND above weekly trend AND daily bullish
        # Short: breakout down with volume AND below weekly trend AND daily bearish
        long_setup = breakout_up and volume_confirmed and above_weekly_trend and daily_bull
        short_setup = breakout_down and volume_confirmed and below_weekly_trend and daily_bear
        
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

</think>