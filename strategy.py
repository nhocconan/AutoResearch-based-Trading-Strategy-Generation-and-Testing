#!/usr/bin/env python3
"""
Experiment #134: 1h Donchian Breakout + Volume Spike + 4h/1d Regime Filter

HYPOTHESIS: Donchian breakouts capture momentum bursts. Volume spike confirms institutional participation.
4h/1d regime filter (ADX + Chop) ensures we only trade breakouts in favorable market conditions.
Primary timeframe: 1h for entry timing. HTF: 4h for trend, 1d for regime.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
Uses session filter (08-20 UTC) to avoid low-liquidity periods. Fixed size 0.20.
Works in bull/bear: in strong trends (ADX>25) we follow breakouts; in chop (CHOP>61.8) we avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_vol_4h1d_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === HTF: 1d data for regime filter (ADX, Chop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ADX on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    # Calculate Chopiness Index on 1d data
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period has no previous close
        
        # Sum of TR over period
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        # Chop = LOG10(tr_sum / (hh - ll)) / LOG10(period) * 100
        # Avoid division by zero
        hl_range = hh - ll
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if hl_range[i] > 0 and not np.isnan(tr_sum[i]) and tr_sum[i] > 0:
                chop[i] = np.log10(tr_sum[i] / hl_range[i]) / np.log10(period) * 100
            else:
                chop[i] = 50.0  # neutral when range is zero
        return chop
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators: Donchian Channel (20) for trend ===
    def donchian_channel(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_4h_upper, dc_4h_lower = donchian_channel(high_4h, low_4h, 20)
    dc_4h_upper_aligned = align_htf_to_ltf(prices, df_4h, dc_4h_upper)
    dc_4h_lower_aligned = align_htf_to_ltf(prices, df_4h, dc_4h_lower)
    
    # === 1h Indicators: Donchian Breakout + Volume Spike ===
    # Donchian Channel (20) on 1h
    dc_1h_upper, dc_1h_lower = donchian_channel(high, low, 20)
    
    # Volume Spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # === Session Filter (08-20 UTC) ===
    # open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Fixed position size (20% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Skip if outside trading session ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Skip if any data is invalid ---
        if (np.isnan(dc_1h_upper[i]) or np.isnan(dc_1h_lower[i]) or 
            np.isnan(dc_4h_upper_aligned[i]) or np.isnan(dc_4h_lower_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        adx_val = adx_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        is_trending = adx_val > 25
        is_chop = chop_val > 61.8  # Choppy market
        
        # Avoid trading in choppy markets unless strong trend
        if is_chop and not is_trending:
            signals[i] = 0.0
            continue
        
        # --- Breakout Conditions ---
        # Long breakout: price > 1h Donchian upper + volume spike
        long_breakout = (close[i] > dc_1h_upper[i]) and volume_spike[i]
        # Short breakout: price < 1h Donchian lower + volume spike
        short_breakout = (close[i] < dc_1h_lower[i]) and volume_spike[i]
        
        # --- HTF Trend Filter (4h Donchian) ---
        # Only long if price above 4h Donchian middle (bullish bias)
        # Only short if price below 4h Donchian middle (bearish bias)
        dc_4h_middle = (dc_4h_upper_aligned[i] + dc_4h_lower_aligned[i]) / 2
        long_trend_filter = close[i] > dc_4h_middle
        short_trend_filter = close[i] < dc_4h_middle
        
        # --- Generate Signals ---
        if long_breakout and long_trend_filter:
            signals[i] = SIZE
        elif short_breakout and short_trend_filter:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals