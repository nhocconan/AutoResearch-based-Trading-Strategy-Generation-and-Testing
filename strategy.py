#!/usr/bin/env python3
"""
Experiment #134: 1h Donchian Breakout + 4h/1d Regime Filter

HYPOTHESIS: Use 4h/1d timeframes for trend and regime detection (ADX + chop), 
1h for precise Donchian breakout entries with volume confirmation.
In trending regimes (ADX>25): trade breakouts in trend direction.
In choppy regimes (CHOP>61.8): fade breakouts (mean reversion).
Session filter (08-20 UTC) reduces noise. Target: 60-150 trades over 4 years.
Works in bull/bear via regime adaptation: trend following in strong trends, 
mean reversion in ranges.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_4h_1d_regime_v1"
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
    
    # === HTF: 4h data for trend direction (EMA21) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for regime filters (ADX + Chop) ===
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
        atr = np.zeros_like(high)
        for i in range(len(high)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1] if i>0 else close[i]), abs(low[i] - close[i-1] if i>0 else close[i]))
        
        # Sum of ATR over period
        sum_atr = np.zeros_like(atr)
        for i in range(period-1, len(atr)):
            if i == period-1:
                sum_atr[i] = np.sum(atr[i-period+1:i+1])
            else:
                sum_atr[i] = sum_atr[i-1] - atr[i-period] + atr[i]
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(len(high)):
            if i < period-1:
                highest_high[i] = np.max(high[:i+1])
                lowest_low[i] = np.min(low[:i+1])
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop formula: 100 * log10(sum(ATR)/log(highest_high - lowest_low)) / log10(period)
        range_hl = highest_high - lowest_low
        chop = np.zeros_like(atr)
        for i in range(period-1, len(atr)):
            if range_hl[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral when range is zero
        
        return chop
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 1h Indicators: Donchian Channels (20-period) ===
    def donchian_channels(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(len(high)):
            if i < period-1:
                upper[i] = np.max(high[:i+1])
                lower[i] = np.min(low[:i+1])
            else:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    # Volume spike detector (20-period)
    def volume_spike(vol, period=20, threshold=2.0):
        avg_vol = np.zeros_like(vol)
        for i in range(len(vol)):
            if i < period-1:
                avg_vol[i] = np.mean(vol[:i+1])
            else:
                avg_vol[i] = np.mean(vol[i-period+1:i+1])
        spike = vol / (avg_vol + 1e-10)
        return spike > threshold
    
    vol_spike = volume_spike(volume, 20, 2.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter (08-20 UTC) ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if in_position:
                # Check stoploss: 2*ATR against position
                # Approximate ATR with 20-period range
                atr_approx = (donch_upper[i] - donch_lower[i]) / 4  # rough ATR proxy
                if position_side > 0 and close[i] < entry_price - 2.0 * atr_approx:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif position_side < 0 and close[i] > entry_price + 2.0 * atr_approx:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_side * SIZE
            else:
                signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i])):
            if in_position:
                signals[i] = position_side * SIZE
            else:
                signals[i] = 0.0
            continue
        
        # --- Regime Detection ---
        adx_val = adx_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        is_trending = adx_val > 25
        is_choppy = chop_val > 61.8
        
        # --- Donchian Breakout Signals ---
        breakout_up = close[i] > donch_upper[i-1] and vol_spike[i]
        breakout_down = close[i] < donch_lower[i-1] and vol_spike[i]
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Stoploss: 2*ATR approximation
            atr_approx = (donch_upper[i] - donch_lower[i]) / 4
            if position_side > 0 and close[i] < entry_price - 2.0 * atr_approx:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            elif position_side < 0 and close[i] > entry_price + 2.0 * atr_approx:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            # Time-based exit: max 24 hours (24 bars on 1h)
            elif i - entry_bar >= 24:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if breakout_up or breakout_down:
            if is_trending:
                # Trend regime: follow breakout direction
                if breakout_up and close[i] > ema_4h_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_bar = i
                    signals[i] = SIZE
                elif breakout_down and close[i] < ema_4h_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_bar = i
                    signals[i] = -SIZE
            elif is_choppy:
                # Choppy regime: fade breakouts (mean reversion)
                if breakout_up:
                    in_position = True
                    position_side = -1  # short on upward breakout
                    entry_price = close[i]
                    entry_bar = i
                    signals[i] = -SIZE
                elif breakout_down:
                    in_position = True
                    position_side = 1   # long on downward breakout
                    entry_price = close[i]
                    entry_bar = i
                    signals[i] = SIZE
    
    return signals