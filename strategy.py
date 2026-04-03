#!/usr/bin/env python3
"""
Experiment #234: 1h Donchian(20) Breakout + 4h/1d Trend Filter + Volume Spike

HYPOTHESIS: 1h Donchian breakouts filtered by 4h EMA trend and 1d pivot direction reduce false signals. 
Volume spike (>2.0x average) confirms momentum. Using 4h/1d for signal direction and 1h only for entry timing 
targets 15-37 trades/year (60-150 total over 4 hours) to minimize fee drag. Works in bull markets (breakouts 
with volume) and bear markets (failed breaks reverse sharply). ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_234_1h_donchian_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 21:
        close_4h = df_4h['close'].values.astype(np.float64)
        ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1)
        trend_4h_up = ema_4h_aligned > np.roll(ema_4h_aligned, 1)  # EMA rising
        trend_4h_down = ema_4h_aligned < np.roll(ema_4h_aligned, 1)  # EMA falling
    else:
        ema_4h_aligned = np.full(n, np.nan)
        trend_4h_up = np.zeros(n, dtype=bool)
        trend_4h_down = np.zeros(n, dtype=bool)
    
    # === HTF: 1d data for daily pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        df_1d_indexed = df_1d.set_index('open_time')
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        prior_day_pivot = (prior_day_high + prior_day_low + prior_day_close) / 3.0
        daily_pivot_series = pd.Series(index=df_1d_indexed.index, data=prior_day_pivot)
        daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_series.values)
    else:
        daily_pivot_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Ensure enough data for HTF EMA, daily pivot, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(daily_pivot_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade during 08-20 UTC ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- 4h Trend Filter: EMA direction ---
        price_above_ema_4h = close[i] > ema_4h_aligned[i]
        price_below_ema_4h = close[i] < ema_4h_aligned[i]
        
        # --- Daily Pivot Filter: Price > pivot = bullish bias, Price < pivot = bearish bias ---
        price_above_daily_pivot = close[i] > daily_pivot_aligned[i]
        price_below_daily_pivot = close[i] < daily_pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- 1h Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above 4h EMA + price above daily pivot
        long_condition = breakout_up and volume_spike and price_above_ema_4h and price_above_daily_pivot
        
        # Short: Donchian breakout down + volume spike + price below 4h EMA + price below daily pivot
        short_condition = breakout_down and volume_spike and price_below_ema_4h and price_below_daily_pivot
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals