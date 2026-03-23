#!/usr/bin/env python3
"""
Experiment #179: 4h Primary + 1d HTF — Connors RSI + Donchian Breakout + Choppiness Regime

Hypothesis: Previous 12h strategies failed due to too few trades. 4h timeframe with
Connors RSI (proven on ETH with Sharpe +0.923) combined with Donchian breakouts and
Choppiness regime filter should generate 25-45 trades/year with positive Sharpe on
ALL symbols. Key insight: use OR logic for entries (not AND) to ensure trades fire.

KEY IMPROVEMENTS:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Donchian(20) breakout for trend confirmation
3. Choppiness Index for regime-aware sizing (not entry block)
4. 1d HMA for macro bias (HTF trend filter)
5. Looser entry: CRSI<20 OR Donchian breakout (not both required)
6. Position size: 0.30 full, 0.20 partial based on regime
7. ATR trailing stop at 2.5x

TARGET: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_chop_regime_1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10-20, Short when CRSI > 80-90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for mean reversion
    rsi_fast = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 / (1.0 + streak_period / max(streak_abs[i], 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 / (1.0 + streak_period / max(streak_abs[i], 1))
            streak_rsi[i] = 100.0 - streak_rsi[i]
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - where current price ranks in last N periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # CRSI = average of three components
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = np.maximum(high_s - high_s.shift(1), 0).values
    minus_dm = np.maximum(low_s.shift(1) - low, 0).values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = np.nan_to_num(adx, nan=0.0)
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(adx[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === REGIME DETECTION ===
        chop_value = chop_14[i]
        is_trending = chop_value < 45.0
        is_ranging = chop_value > 55.0
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CRSI EXTREMES (mean reversion) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 20.0
        adx_weak = adx[i] < 25.0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLUME FILTER (lenient) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === ENTRY LOGIC (OR logic to ensure trades) ===
        new_signal = 0.0
        
        # LONG entries - multiple independent triggers (any one can fire)
        long_trigger_1 = crsi_oversold and price_above_hma_1d  # Mean reversion + bullish bias
        long_trigger_2 = breakout_long and price_above_hma_1d  # Breakout + bullish bias
        long_trigger_3 = rsi_oversold and adx_weak  # RSI oversold in range
        
        if long_trigger_1 or long_trigger_2 or long_trigger_3:
            if is_trending and price_above_hma_1d:
                new_signal = POSITION_SIZE_FULL
            elif is_ranging:
                new_signal = POSITION_SIZE_HALF
            else:
                new_signal = POSITION_SIZE_HALF
        
        # SHORT entries - multiple independent triggers (any one can fire)
        short_trigger_1 = crsi_overbought and price_below_hma_1d  # Mean reversion + bearish bias
        short_trigger_2 = breakout_short and price_below_hma_1d  # Breakout + bearish bias
        short_trigger_3 = rsi_overbought and adx_weak  # RSI overbought in range
        
        if short_trigger_1 or short_trigger_2 or short_trigger_3:
            if is_trending and price_below_hma_1d:
                new_signal = -POSITION_SIZE_FULL
            elif is_ranging:
                new_signal = -POSITION_SIZE_HALF
            else:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1d HMA
                if price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1d HMA
                if price_below_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d HMA significantly
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d HMA significantly
        if in_position and position_side < 0 and price_above_hma_1d:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals