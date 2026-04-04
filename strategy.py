#!/usr/bin/env python3
"""
Experiment #4331: 6h Elder Ray + 1d Regime Filter + Volume Spike
HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h captures institutional buying/selling pressure when aligned with 1d trend regime (ADX>25) and confirmed by volume spikes (>2.0x average). Works in bull via strong Bull Power breakouts, in bear via Bear Power breakdowns. ADX regime filter prevents whipsaw in ranging markets. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4331_6h_elder_ray_1d_adx_vol_v1"
timeframe = "6h"
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
    
    # === Precompute HTF: 1d ADX for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # Calculate ADX components
        plus_dm = np.zeros(len(df_1d))
        minus_dm = np.zeros(len(df_1d))
        tr = np.zeros(len(df_1d))
        
        for i in range(1, len(df_1d)):
            high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
            low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
            
            plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
            minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
            
            tr[i] = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        period = 14
        alpha = 1.0 / period
        
        atr_1d = np.zeros(len(df_1d))
        atr_1d[period-1] = np.nanmean(tr[period-1:2*period-1]) if len(tr) >= 2*period-1 else np.nan
        
        plus_dm_smooth = np.zeros(len(df_1d))
        minus_dm_smooth = np.zeros(len(df_1d))
        
        if len(df_1d) >= period:
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[period-1:2*period-1])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[period-1:2*period-1])
            
            for i in range(period, len(df_1d)):
                atr_1d[i] = atr_1d[i-1] * (1 - alpha) + alpha * tr[i]
                plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + alpha * plus_dm[i]
                minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + alpha * minus_dm[i]
        
        # Avoid division by zero
        plus_di_1d = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
        minus_di_1d = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
        dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                         100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
        
        adx_1d = np.full(len(df_1d), np.nan)
        if len(df_1d) >= 2*period-1:
            adx_1d[2*period-2] = np.nanmean(dx_1d[period-1:2*period-1])
            for i in range(2*period-1, len(df_1d)):
                adx_1d[i] = adx_1d[i-1] * (1 - alpha) + alpha * dx_1d[i]
        
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(13, 20, 14, 2*14)  # EMA13, vol MA, ATR, ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending_regime = adx_1d_aligned[i] > 25.0
        
        if volume_confirm and trending_regime:
            # Elder Ray signals
            bull_breakout = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]  # Rising bull power
            bear_breakout = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]  # Falling bear power (more negative)
            
            # Long conditions: Rising Bull Power + volume + trend
            long_entry = bull_breakout
            
            # Short conditions: Falling Bear Power + volume + trend
            short_entry = bear_breakout
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals