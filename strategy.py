#!/usr/bin/env python3
"""
Experiment #208: 12h Donchian(20) Breakout + Weekly Trend Filter + Volume Spike

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by weekly trend direction 
(using weekly EMA crossover: EMA20 > EMA50 = bullish bias, EMA20 < EMA50 = bearish bias) 
and confirmed by volume spikes (>2.0x average), capture strong momentum moves with reduced 
false breakouts. Weekly EMA provides structural trend from higher timeframe (1w/1d), 
aligning with institutional order flow. 12h timeframe targets 12-37 trades/year (50-150 
total over 4 years) to minimize fee drag while capturing significant moves. Volume 
confirmation filters out low-conviction breakouts. ATR-based stoploss manages risk.
Works in both bull (breakouts with trend) and bear (breakouts against trend filtered out) 
markets by requiring alignment with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_208_12h_donchian_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly EMA calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA20 and EMA50 from 1d data
    weekly_ema20 = np.full(n, np.nan)
    weekly_ema50 = np.full(n, np.nan)
    
    if len(df_1d) >= 50:  # Need enough data for EMA50
        # Create daily index for resampling
        df_1d_indexed = df_1d.copy()
        df_1d_indexed.index = pd.date_range(
            start=df_1d['open_time'].iloc[0], 
            periods=len(df_1d), 
            freq='1d'
        )
        
        # Resample 1d to weekly using actual weekly boundaries
        weekly_close = df_1d_indexed['close'].resample('W').last()
        weekly_high = df_1d_indexed['high'].resample('W').max()
        weekly_low = df_1d_indexed['low'].resample('W').min()
        
        if len(weekly_close) >= 2:  # Need at least 2 weeks
            # Calculate EMAs on weekly close
            ema20_weekly = pd.Series(weekly_close.values).ewm(span=20, min_periods=20, adjust=False).mean().values
            ema50_weekly = pd.Series(weekly_close.values).ewm(span=50, min_periods=50, adjust=False).mean().values
            
            # Map back to daily frequency
            daily_ema20 = pd.Series(index=df_1d_indexed.index, dtype=np.float64)
            daily_ema50 = pd.Series(index=df_1d_indexed.index, dtype=np.float64)
            
            for i in range(len(weekly_close)):
                week_start = weekly_close.index[i]
                week_end = weekly_close.index[i+1] if i+1 < len(weekly_close) else weekly_close.index[-1] + pd.Timedelta(days=7)
                if i < len(ema20_weekly) and not np.isnan(ema20_weekly[i]):
                    daily_ema20.loc[week_start:week_end] = ema20_weekly[i]
                if i < len(ema50_weekly) and not np.isnan(ema50_weekly[i]):
                    daily_ema50.loc[week_start:week_end] = ema50_weekly[i]
            
            # Align to 12h timeframe
            daily_ema20_values = daily_ema20.reindex(df_1d_indexed.index).values
            daily_ema50_values = daily_ema50.reindex(df_1d_indexed.index).values
            
            weekly_ema20_aligned = align_htf_to_ltf(prices, df_1d, daily_ema20_values)
            weekly_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50_values)
        else:
            weekly_ema20_aligned = np.full(n, np.nan)
            weekly_ema50_aligned = np.full(n, np.nan)
    else:
        weekly_ema20_aligned = np.full(n, np.nan)
        weekly_ema50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF weekly EMA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(weekly_ema50_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Trend Filter: EMA20 > EMA50 = bullish bias, EMA20 < EMA50 = bearish bias ---
        weekly_bullish = weekly_ema20_aligned[i] > weekly_ema50_aligned[i]
        weekly_bearish = weekly_ema20_aligned[i] < weekly_ema50_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + weekly bullish trend
        long_condition = breakout_up and volume_spike and weekly_bullish
        
        # Short: Donchian breakout down + volume spike + weekly bearish trend
        short_condition = breakout_down and volume_spike and weekly_bearish
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals