#!/usr/bin/env python3
"""
Experiment #022: 12h Primary + 1d/1w HTF — Dual Regime with Connors RSI + Donchian

Hypothesis: 12h timeframe with daily/weekly trend bias should generate 25-45 trades/year.
Combines proven patterns from experiment history:
- Connors RSI for mean reversion timing (worked on ETH with Sharpe +0.923)
- Donchian breakout for trend confirmation (worked on SOL with Sharpe +0.782)
- Choppiness Index for regime detection (essential for bear/range markets)
- 1d HMA for intermediate trend bias
- 1w HMA for macro trend filter

Key insight: Use LOOSE entry thresholds to guarantee trade generation while
maintaining edge through multi-timeframe confluence and regime adaptation.

Why 12h should work:
- Fewer trades than 1h/4h = less fee drag
- Still responsive enough to catch major moves
- 1d/1w HTF provides strong trend confirmation

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_donchian_chop_regime_1d1w_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    This is a proven mean-reversion indicator with ~75% win rate.
    Long when CRSI < 10, Short when CRSI > 90.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calculate_rsi(streak, period=streak_period)
    
    # Percent Rank component
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        pct_rank[i] = rank / rank_period * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=n, min_periods=n, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=n, min_periods=n, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=n, min_periods=n, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100.0 * plus_dm_s / (atr + 1e-10)
    minus_di = 100.0 * minus_dm_s / (atr + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=n, min_periods=n, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Also calculate price momentum
    roc_10 = np.zeros(n)
    for i in range(10, n):
        roc_10[i] = (close[i] - close[i-10]) / (close[i-10] + 1e-10) * 100.0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(120, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(adx_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # LOOSE threshold for more trades
        is_trending = chop_value < 45.0  # LOOSE threshold for more trades
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 20.0  # Was 10, now 20 for more longs
        crsi_overbought = crsi[i] > 80.0  # Was 90, now 80 for more shorts
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] * 0.995  # Near upper
        donchian_breakout_short = close[i] < donchian_lower[i-1] * 1.005  # Near lower
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0  # LOOSE threshold
        adx_weak = adx_14[i] < 25.0
        
        # === MOMENTUM FILTER ===
        momentum_positive = roc_10[i] > -2.0  # Not strongly negative
        momentum_negative = roc_10[i] < 2.0  # Not strongly positive
        
        # === VOLATILITY FILTER ===
        vol_elevated = atr_7[i] > atr_14[i] * 1.05  # Recent vol slightly above average
        
        # === ADAPTIVE REGIME ENTRY LOGIC (LOOSE CONDITIONS) ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + weekly bias helps OR daily bias helps
            if crsi_oversold:
                if price_above_hma_1w or price_above_hma_1d or vol_elevated:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + weekly bias helps OR daily bias helps
            elif crsi_overbought:
                if price_below_hma_1w or price_below_hma_1d or vol_elevated:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Donchian ---
        elif is_trending:
            # Long: Donchian breakout + ADX strong + trend confirmation
            if donchian_breakout_long and adx_strong:
                if price_above_hma_1w or price_above_hma_1d:
                    if momentum_positive:
                        new_signal = POSITION_SIZE
            
            # Short: Donchian breakout + ADX strong + trend confirmation
            elif donchian_breakout_short and adx_strong:
                if price_below_hma_1w or price_below_hma_1d:
                    if momentum_negative:
                        new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Simple CRSI extreme if no regime signal ---
        if new_signal == 0.0:
            # Long: Very oversold CRSI
            if crsi[i] < 15.0:
                if price_above_hma_1d or momentum_positive:
                    new_signal = POSITION_SIZE
            
            # Short: Very overbought CRSI
            elif crsi[i] > 85.0:
                if price_below_hma_1d or momentum_negative:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if both weekly and daily trend turn bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if both weekly and daily trend turn bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and price_above_hma_1d:
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