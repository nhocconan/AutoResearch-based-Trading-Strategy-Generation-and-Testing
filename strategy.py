#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian Breakout + Volume Spike + Choppiness Regime

HYPOTHESIS: Simple price-channel breakout strategy based on PROVEN DB winners.
The DB shows Donchian(20) + volume + chop regime → SOL test Sharpe 1.10-1.38.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Price breaks above 20-bar high + volume confirms = strong momentum long
- Bear: Price breaks below 20-bar low + volume confirms = strong momentum short
- Range (CHOP > 61.8): No trades = avoids whipsaws during consolidation
- Symmetric entry/exit = works in both directions

KEY INSIGHT: Simplicity wins. The best test performers (Sharpe 1.3-1.8) use
ONE strong signal (Donchian breakout) + volume confirmation + regime filter.
Previous failed strategies had too many conditions = too few trades.

TARGET: 100-250 total trades over 4 years (25-62/year) on 4h.
Keep rate target: 40-50% (DB average for 4h strategies).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_v6"
timeframe = "4h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20-bar breakout system"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = CHOPPY (range-bound, no trend)
    CHOP < 38.2 = TRENDING (trend-following environment)
    Used to filter out trades during consolidation
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if tr_sum > 0 and hh > ll:
            # CHOP = 100 * log10(tr_sum / (hh - ll)) / log10(period)
            chop[i] = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
                adx[i] = dx
    
    adx_smooth = pd.Series(adx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx_smooth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF: 1d EMA for trend direction ===
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # 1d EMA(21) for trend
    ema_1d = pd.Series(df_1d_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # 1d Donchian for structure
    htf_donch_upper, htf_donch_lower, _ = calculate_donchian(df_1d_high, df_1d_low, period=20)
    
    # HTF: Price above EMA = bull, below = bear
    htf_bullish = df_1d_close > ema_1d
    htf_bearish = df_1d_close < ema_1d
    
    # Align HTF to 4h
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    htf_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d_close)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # ADX for additional confirmation
    adx = calculate_adx(high, low, close, period=14)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 50  # Donchian needs 20 bars
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK: Only trade in trending or neutral market ===
        # CHOP > 61.8 = very choppy, skip
        # CHOP < 50 = trending, allow trades
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # ADX confirmation (trend strength)
        strong_trend = adx[i] > 22 if not np.isnan(adx[i]) else False
        
        # === HTF TREND ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else False
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Price closes above 20-bar high = bullish breakout
        bull_breakout = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else close[i] > donch_upper[i]
        
        # Price closes below 20-bar low = bearish breakout
        bear_breakout = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume confirm + (trending OR strong trend)
            # Skip if choppy
            if not is_choppy and (is_trending or strong_trend):
                if bull_breakout and vol_spike:
                    # Prefer HTF bull, but allow neutral
                    if htf_bull or not htf_bear:
                        desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume confirm + (trending OR strong trend)
            if not is_choppy and (is_trending or strong_trend):
                if bear_breakout and vol_spike:
                    # Prefer HTF bear, but allow neutral
                    if htf_bear or not htf_bull:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
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