#!/usr/bin/env python3
"""
Experiment #010: 4h Regime-Adaptive Strategy with 1d HTF Trend + Choppiness Filter
Hypothesis: 4h timeframe captures multi-day swings with less noise than 15m/30m.
Using Choppiness Index to detect regime (trend vs range), then apply appropriate logic:
- Trending (CHOP<38): Donchian breakout + 1d HMA alignment
- Ranging (CHOP>61): RSI/Bollinger mean reversion
- Neutral: Stay flat or reduce position
This adaptive approach should work in both 2021-2024 bull/bear and 2025 range markets.
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.25-0.30).
Timeframe: 4h (REQUIRED), HTF: 1d for major trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_1d_hma_donchian_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(er_period, n):
        change[i] = np.abs(close[i] - close[i-er_period])
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc[0] = slow_sc  # default for first valid
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    
    # 4h HMA for additional trend confirmation
    hma_4h = calculate_hma(close, 21)
    hma_4h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - major trend direction
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h trend
        hma_4h_bullish = close[i] > hma_4h[i]
        hma_4h_bearish = close[i] < hma_4h[i]
        hma_rising = hma_4h[i] > hma_4h[i-5] if i > 4 else False
        hma_falling = hma_4h[i] < hma_4h[i-5] if i > 4 else False
        
        # Regime detection via Choppiness Index
        regime_trending = chop[i] < 45  # trending market
        regime_ranging = chop[i] > 55   # ranging market
        
        # ADX confirmation for trend strength
        adx_strong = adx[i] > 20
        adx_weak = adx[i] < 25
        
        # Donchian breakout signals
        breakout_long = close[i] > donch_upper[i-2] if i > 1 else False
        breakout_short = close[i] < donch_lower[i-2] if i > 1 else False
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Bollinger Band extremes
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_rising = kama[i] > kama[i-5] if i > 4 else False
        kama_falling = kama[i] < kama[i-5] if i > 4 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_4h_fast[i] > hma_4h[i]
        fast_below_slow = hma_4h_fast[i] < hma_4h[i]
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 45) - Trend Following ===
        if regime_trending and adx_strong:
            # Long: Donchian breakout + HTF bullish + 4h bullish
            if breakout_long and htf_bullish and hma_4h_bullish:
                new_signal = SIZE_ENTRY
            
            # Long: KAMA rising + HTF bullish + Fast HMA crossover
            elif kama_bullish and kama_rising and htf_bullish and fast_above_slow:
                new_signal = SIZE_ENTRY
            
            # Short: Donchian breakdown + HTF bearish + 4h bearish
            elif breakout_short and htf_bearish and hma_4h_bearish:
                new_signal = -SIZE_ENTRY
            
            # Short: KAMA falling + HTF bearish + Fast HMA crossover down
            elif kama_bearish and kama_falling and htf_bearish and fast_below_slow:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (CHOP > 55) - Mean Reversion ===
        elif regime_ranging or adx_weak:
            # Long: RSI oversold + price near BB lower + HTF not strongly bearish
            if rsi_oversold and bb_oversold and not htf_bearish:
                new_signal = SIZE_ENTRY
            
            # Long: BB lower touch + 4h HMA support
            elif bb_oversold and close[i] > hma_4h[i] * 0.98:
                new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + price near BB upper + HTF not strongly bullish
            if rsi_overbought and bb_overbought and not htf_bullish:
                new_signal = -SIZE_ENTRY
            
            # Short: BB upper touch + 4h HMA resistance
            elif bb_overbought and close[i] < hma_4h[i] * 1.02:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME (38 < CHOP < 62) - Reduced Signals ===
        else:
            # Only enter on strong confirmations
            if breakout_long and htf_bullish and rsi[i] < 60:
                new_signal = SIZE_ENTRY * 0.7
            
            if breakout_short and htf_bearish and rsi[i] > 40:
                new_signal = -SIZE_ENTRY * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals