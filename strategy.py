#!/usr/bin/env python3
"""
Experiment #025: 15m Adaptive Regime Strategy with 4h/1h MTF Filters
Hypothesis: 15m timeframe captures intraday swings but needs strong HTF filters to avoid noise.
4h HMA provides major trend regime (bull/bear). 1h RSI confirms momentum direction.
Choppiness Index (CHOP) detects range vs trend regimes to switch strategy mode.
In trending regime (CHOP<38): follow Supertrend with HTF confirmation.
In ranging regime (CHOP>62): mean revert on RSI extremes with tight stops.
Multiple entry triggers ensure ≥10 trades per symbol. Position size 0.25-0.30 with 2.5x ATR stop.
This adaptive approach should work across bull (2021), bear (2022), and range (2025) markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_adaptive_regime_4h_1h_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop[:period] = 50.0
    return np.clip(chop, 0, 100)

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for major trend regime
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h RSI for momentum confirmation
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 2.5)
    chop = calculate_choppiness(high, low, close, 14)
    
    # Donchian channels for breakout detection
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    
    # 15m HMA for short-term trend
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 9)
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # === REGIME DETECTION ===
        # 4h trend regime (major direction)
        hma_4h_valid = hma_4h_aligned[i] > 0
        regime_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        regime_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h momentum confirmation
        rsi_1h_bullish = rsi_1h_aligned[i] > 45 and rsi_1h_aligned[i] < 70
        rsi_1h_bearish = rsi_1h_aligned[i] > 30 and rsi_1h_aligned[i] < 55
        
        # Choppiness regime (trend vs range)
        trending_regime = chop[i] < 45  # Clear trend
        ranging_regime = chop[i] > 55   # Clear range
        neutral_regime = not trending_regime and not ranging_regime
        
        # Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip (strongest signal)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # HMA crossover on 15m
        hma_cross_long = hma_15m_fast[i] > hma_15m[i] and hma_15m_fast[i-1] <= hma_15m[i-1]
        hma_cross_short = hma_15m_fast[i] < hma_15m[i] and hma_15m_fast[i-1] >= hma_15m[i-1]
        
        # Donchian breakout
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # RSI extremes on 15m (for mean reversion)
        rsi_oversold = rsi_15m[i] < 30
        rsi_overbought = rsi_15m[i] > 70
        rsi_neutral = rsi_15m[i] > 40 and rsi_15m[i] < 60
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # === ENTRY LOGIC - MULTIPLE TRIGGERS ===
        new_signal = 0.0
        
        # --- LONG ENTRY TRIGGERS ---
        # Trigger 1: Supertrend flip + 4h bullish (trend following in bullish regime)
        if st_flip_long and regime_bullish:
            new_signal = SIZE
        
        # Trigger 2: Supertrend flip + 1h RSI bullish (momentum confirmation)
        elif st_flip_long and rsi_1h_bullish:
            new_signal = SIZE
        
        # Trigger 3: Trending regime + Supertrend long + HMA cross (trend continuation)
        elif trending_regime and st_long and hma_cross_long and regime_bullish:
            new_signal = SIZE
        
        # Trigger 4: Ranging regime + RSI oversold + price near Donchian lower (mean reversion)
        elif ranging_regime and rsi_oversold and close[i] < donch_mid[i]:
            new_signal = SIZE
        
        # Trigger 5: Donchian breakout + volume + 4h bullish (breakout with confirmation)
        elif donch_breakout_long and vol_confirm and regime_bullish:
            new_signal = SIZE
        
        # Trigger 6: HMA cross long + Supertrend long + RSI rising (momentum entry)
        elif hma_cross_long and st_long and rsi_15m[i] > rsi_15m[i-3] if i > 3 else False:
            if rsi_15m[i] > 45:
                new_signal = SIZE
        
        # --- SHORT ENTRY TRIGGERS ---
        # Trigger 1: Supertrend flip + 4h bearish (trend following in bearish regime)
        if st_flip_short and regime_bearish:
            new_signal = -SIZE
        
        # Trigger 2: Supertrend flip + 1h RSI bearish (momentum confirmation)
        elif st_flip_short and rsi_1h_bearish:
            new_signal = -SIZE
        
        # Trigger 3: Trending regime + Supertrend short + HMA cross (trend continuation)
        elif trending_regime and st_short and hma_cross_short and regime_bearish:
            new_signal = -SIZE
        
        # Trigger 4: Ranging regime + RSI overbought + price near Donchian upper (mean reversion)
        elif ranging_regime and rsi_overbought and close[i] > donch_mid[i]:
            new_signal = -SIZE
        
        # Trigger 5: Donchian breakout + volume + 4h bearish (breakout with confirmation)
        elif donch_breakout_short and vol_confirm and regime_bearish:
            new_signal = -SIZE
        
        # Trigger 6: HMA cross short + Supertrend short + RSI falling (momentum entry)
        elif hma_cross_short and st_short and rsi_15m[i] < rsi_15m[i-3] if i > 3 else False:
            if rsi_15m[i] < 55:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        atr_mult = 2.5
        
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - atr_mult * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] - atr_mult * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if trailing_stop > 0 and close[i] < trailing_stop:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price + 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + atr_mult * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] + atr_mult * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if trailing_stop > 0 and close[i] > trailing_stop:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price - 2.5 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - atr_mult * atr[i] if position_side > 0 else close[i] + atr_mult * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - atr_mult * atr[i] if position_side > 0 else close[i] + atr_mult * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals