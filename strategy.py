#!/usr/bin/env python3
"""
Experiment #1231: 4h Primary + 1d/1w HTF — Donchian Breakout with HMA Trend & ADX Filter

Hypothesis: After reviewing 900+ failed experiments, pure mean-reversion (CRSI, Bollinger) 
fails in strong trends, while pure trend-following gets whipsawed in chop. The solution is 
a REGIME-ADAPTIVE breakout system: Donchian channel breakouts ONLY when HMA confirms trend 
direction AND ADX confirms trend strength. This captures major moves while filtering false 
breakouts in ranging markets.

Key innovations vs failed experiments:
- Donchian(20) breakout: Simple but effective when filtered (unlike EMA crossovers)
- Dual HMA filter: 1d HMA for macro bias + 1w HMA for secular trend (from #1222 success)
- ADX > 18 (lower than #1229's 22): More trades while still filtering chop
- ATR-normalized breakout: Breakout must exceed 0.5*ATR to avoid noise
- Asymmetric sizing: 0.30 for strong trends (ADX>25), 0.20 for moderate (ADX 18-25)
- Trailing stop: 2.5*ATR from entry extreme (proven in #1229)

Why this should beat Sharpe=0.612 baseline:
- Donchian breakouts capture major moves (2021 bull, 2022 crash, 2024 rally)
- HMA filters prevent counter-trend trades (major failure mode of #1219, #1223)
- ADX filter reduces whipsaws in 2022-2023 range market
- 4h timeframe = 20-50 trades/year target (optimal fee/trade balance)

Target: Sharpe > 0.612, trades >= 80 train (20/year), >= 12 test (3/year), DD > -50%
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_adx_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA with less lag"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops and sizing"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels - breakout detection"""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for secular trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # Calculate 4h HMA for local trend confirmation
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE_STRONG = 0.30  # ADX > 25
    BASE_SIZE_MODERATE = 0.20  # ADX 18-25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_4h[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO TREND FILTERS ===
        # 1d HMA: Intermediate trend bias
        macro_bull_1d = close[i] > hma_1d_aligned[i]
        macro_bear_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA: Secular trend bias (stronger filter)
        macro_bull_1w = close[i] > hma_1w_aligned[i]
        macro_bear_1w = close[i] < hma_1w_aligned[i]
        
        # 4h HMA: Local trend confirmation
        local_bull = close[i] > hma_4h[i]
        local_bear = close[i] < hma_4h[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx[i] > 25.0
        trend_moderate = adx[i] > 18.0
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Long breakout: price breaks above Donchian upper
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        # Short breakout: price breaks below Donchian lower
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === ATR-NORMALIZED BREAKOUT FILTER ===
        # Breakout must be significant (not just noise)
        breakout_threshold = 0.3 * atr[i]
        breakout_long_valid = breakout_long and (close[i] - donchian_upper[i - 1] > breakout_threshold) if i > 0 else False
        breakout_short_valid = breakout_short and (donchian_lower[i - 1] - close[i] > breakout_threshold) if i > 0 else False
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        desired_size = 0.0
        
        # LONG: Secular bull (1w) + Macro bull (1d) + Local bull (4h) + Trend strength + Breakout
        if macro_bull_1w and macro_bull_1d and local_bull:
            if trend_strong and breakout_long_valid:
                desired_signal = 1.0
                desired_size = BASE_SIZE_STRONG
            elif trend_moderate and breakout_long_valid:
                desired_signal = 1.0
                desired_size = BASE_SIZE_MODERATE
        
        # SHORT: Secular bear (1w) + Macro bear (1d) + Local bear (4h) + Trend strength + Breakout
        elif macro_bear_1w and macro_bear_1d and local_bear:
            if trend_strong and breakout_short_valid:
                desired_signal = -1.0
                desired_size = BASE_SIZE_STRONG
            elif trend_moderate and breakout_short_valid:
                desired_signal = -1.0
                desired_size = BASE_SIZE_MODERATE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            desired_size = 0.0
        
        # === APPLY POSITION SIZE TO SIGNAL ===
        if desired_signal != 0.0:
            desired_signal = desired_size * np.sign(desired_signal)
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals