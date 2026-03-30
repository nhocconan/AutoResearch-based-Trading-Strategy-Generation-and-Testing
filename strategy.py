#!/usr/bin/env python3
"""
Experiment #027: Weekly Trend + Daily RSI-Donchian Volume Breakout (12h)

HYPOTHESIS: Simple 2-condition entries that WILL trigger across all markets:
1. Weekly SMA alignment for trend direction (bull/bear/neutral)
2. Daily Donchian(20) breakout + RSI(14) confirmation + volume spike for entries
3. 2.5x ATR trailing stop for risk management

WHY IT WORKS IN BULL AND BEAR:
- Bull: Weekly SMA confirms uptrend → wait for daily pullback + breakout → long
- Bear: Weekly SMA confirms downtrend → wait for daily rally + breakdown → short
- Range: Neutral weekly → mean-revert on daily bands

KEY INSIGHT: DB winners (Sharpe 1.3-1.5) use price channels + volume + simple regime.
This is the simplest possible version that still captures the pattern.

TARGET: 100-250 total trades over 4 years (25-62/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_weekly_sma_daily_donchian_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(prices, period=14):
    """RSI with proper min_periods"""
    delta = np.diff(prices, prepend=prices[0])
    delta[0] = 0
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # === Weekly SMA for trend direction ===
    sma_8w = pd.Series(df_1w['close'].values).rolling(window=8, min_periods=8).mean().values
    sma_21w = pd.Series(df_1w['close'].values).rolling(window=21, min_periods=21).mean().values
    
    # Weekly trend: price > SMA21 = bull, price < SMA21 = bear
    weekly_bull = df_1w['close'].values > sma_21w
    weekly_bear = df_1w['close'].values < sma_21w
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull.astype(float))
    weekly_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_bear.astype(float))
    
    # Weekly RSI for additional confirmation
    weekly_rsi = calculate_rsi(df_1w['close'].values, period=14)
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_1w, weekly_rsi)
    
    # === Daily indicators for entry signals ===
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Daily Donchian
    daily_donch_upper, daily_donch_mid, daily_donch_lower = calculate_donchian(
        daily_high, daily_low, period=20
    )
    
    # Daily RSI
    daily_rsi = calculate_rsi(daily_close, period=14)
    
    # Align daily indicators to 12h
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, daily_donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, daily_donch_lower)
    daily_rsi_aligned = align_htf_to_ltf(prices, df_1d, daily_rsi)
    
    # Daily close aligned (for breakout detection)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Local Donchian for entry confirmation
    local_donch_upper, _, local_donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ADX for trend strength
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr_local = np.zeros(n)
    tr_local[0] = high[0] - low[0]
    for i in range(1, n):
        tr_local[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_smooth = pd.Series(tr_local).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    for i in range(14, n):
        if atr_smooth[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
    
    adx = pd.Series(np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100).ewm(
        span=14, min_periods=14, adjust=False
    ).mean().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 200
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND FILTER ===
        weekly_bull = weekly_bull_aligned[i] > 0.5 if not np.isnan(weekly_bull_aligned[i]) else False
        weekly_bear = weekly_bear_aligned[i] > 0.5 if not np.isnan(weekly_bear_aligned[i]) else False
        
        # === DAILY DONCHIAN BREAKOUT ===
        if np.isnan(donch_upper_aligned[i]) or np.isnan(daily_rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        daily_breakout_up = close[i] > donch_upper_aligned[i]
        daily_breakout_down = close[i] < donch_lower_aligned[i]
        
        # Daily RSI confirmation
        daily_rsi_val = daily_rsi_aligned[i]
        daily_rsi_bull = daily_rsi_val > 45 if not np.isnan(daily_rsi_val) else False
        daily_rsi_bear = daily_rsi_val < 55 if not np.isnan(daily_rsi_val) else False
        
        # === LOCAL INDICATORS ===
        local_breakout_up = close[i] > local_donch_upper[i]
        local_breakout_down = close[i] < local_donch_lower[i]
        
        vol_spike = vol_ratio[i] > 1.5
        adx_strong = adx[i] > 20
        
        # RSI extremes for entry
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Weekly bull + daily/local breakout + volume + RSI confirmation
            # Simplified: just need weekly trend + breakout + vol spike (not all conditions)
            long_conditions = (
                weekly_bull and 
                (daily_breakout_up or local_breakout_up) and 
                vol_spike
            )
            
            if long_conditions:
                desired_signal = SIZE
            
            # SHORT: Weekly bear + daily/local breakdown + volume
            short_conditions = (
                weekly_bear and 
                (daily_breakout_down or local_breakout_down) and 
                vol_spike
            )
            
            if short_conditions:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bearish
                if weekly_bear:
                    desired_signal = 0.0
                
                # Exit if local breakdown
                if close[i] < local_donch_lower[i]:
                    desired_signal = 0.0
            
            elif position_side < 0:
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bullish
                if weekly_bull:
                    desired_signal = 0.0
                
                # Exit if local breakout
                if close[i] > local_donch_upper[i]:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals