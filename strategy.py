#!/usr/bin/env python3
"""
Experiment #046: 12h Primary + 1d HTF — Regime-Adaptive with Loose Entries

Hypothesis: 12h timeframe has failed repeatedly (Sharpe=0.000 = no trades) because entry
conditions were too strict. This strategy uses LOOSE thresholds to ensure 20-50 trades/year:
1) Choppiness Index: 45/55 thresholds (not 38/61.8) for regime detection
2) Connors RSI: <25/>75 for mean reversion (not <10/>90)
3) RSI: <40/>60 backup filter (not <30/>70)
4) HMA crossover fallback: ensures trades when other signals don't fire
5) 1d HMA for macro bias (loose: just directional, not strict filter)

Why 12h should work now:
- Previous 12h strategies (#036, #037, #042, #043) had Sharpe=0.000 = ZERO TRADES
- This version has MULTIPLE entry paths (regime + fallback + crossover)
- LOOSE thresholds ensure we capture moves in bear/range markets (2025 test period)
- 1d HTF provides trend bias without over-filtering

Position size: 0.27 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 20-50 trades/year on 12h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_hma_loose_1d_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2) - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = sum(1 for j in range(i-streak_period+1, i+1) if streak[j] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    
    # PercentRank(100) - where current price ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = sum(1 for p in window if p < close[i])
        percent_rank[i] = (rank / rank_period) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short.values + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
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
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.27  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):  # Warmup for all indicators (CRSI needs 100 bars)
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(hma_21[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS (LOOSE - just directional) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1d_slope_up = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i > 5 else False
        hma_1d_slope_down = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i > 5 else False
        
        # === CHOPPINESS REGIME (LOOSE thresholds for trade generation) ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 45.0  # LOOSE: range market
        is_trending = chop_value < 50.0  # LOOSE: trend market (overlap for smooth transition)
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # LOOSE: was <10
        crsi_overbought = crsi[i] > 75.0  # LOOSE: was >90
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        crsi_turning_up = crsi_oversold and crsi_rising
        crsi_turning_down = crsi_overbought and crsi_falling
        
        # === STANDARD RSI (LOOSE backup filter) ===
        rsi_oversold = rsi_14[i] < 40.0  # LOOSE: was <30
        rsi_overbought = rsi_14[i] > 60.0  # LOOSE: was >70
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_squeeze = (bb_upper[i] - bb_lower[i]) < (bb_upper[i-20] - bb_lower[i-20]) * 0.8 if i > 20 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === HMA TREND ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        hma_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        hma_cross_up = close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]
        hma_cross_down = close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]
        
        # === HMA FAST/SLOW CROSSOVER ===
        hma_fast_slow_bull = hma_21[i] > hma_48[i] and hma_21[i-1] <= hma_48[i-1]
        hma_fast_slow_bear = hma_21[i] < hma_48[i] and hma_21[i-1] >= hma_48[i-1]
        
        # === ADAPTIVE REGIME ENTRY LOGIC (MULTIPLE PATHS FOR TRADES) ===
        new_signal = 0.0
        
        # --- PATH 1: RANGING REGIME - Mean Reversion with Connors RSI ---
        if is_ranging:
            # Long: CRSI oversold + any confirmation
            if crsi_turning_up:
                if price_above_hma_1d or price_below_bb_lower or rsi_oversold:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + any confirmation
            elif crsi_turning_down:
                if price_below_hma_1d or price_above_bb_upper or rsi_overbought:
                    new_signal = -POSITION_SIZE
            
            # Backup: BB mean reversion
            elif price_below_bb_lower and rsi_oversold and crsi[i] < 40:
                new_signal = POSITION_SIZE
            elif price_above_bb_upper and rsi_overbought and crsi[i] > 60:
                new_signal = -POSITION_SIZE
        
        # --- PATH 2: TRENDING REGIME - Trend Following ---
        elif is_trending:
            # Long: Donchian breakout + trend confirmation
            if donchian_breakout_long:
                if hma_bullish and (price_above_hma_1d or hma_slope_up):
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + trend confirmation
            elif donchian_breakout_short:
                if hma_bearish and (price_below_hma_1d or hma_slope_down):
                    new_signal = -POSITION_SIZE
            
            # Backup: HMA crossover in trend
            elif hma_cross_up and hma_bullish and price_above_hma_1d:
                new_signal = POSITION_SIZE
            elif hma_cross_down and hma_bearish and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # --- PATH 3: FALLBACK - HMA Fast/Slow Crossover (ensures trades) ---
        if new_signal == 0.0:
            # Long: HMA 21 crosses above 48 + daily bias helps
            if hma_fast_slow_bull:
                if price_above_hma_1d or crsi[i] > 40:
                    new_signal = POSITION_SIZE
            
            # Short: HMA 21 crosses below 48 + daily bias helps
            elif hma_fast_slow_bear:
                if price_below_hma_1d or crsi[i] < 60:
                    new_signal = -POSITION_SIZE
        
        # --- PATH 4: ULTIMATE FALLBACK - Simple HMA Cross (guarantees some trades) ---
        if new_signal == 0.0:
            if hma_cross_up and crsi_rising:
                new_signal = POSITION_SIZE * 0.5  # Half size for weaker signal
            elif hma_cross_down and crsi_falling:
                new_signal = -POSITION_SIZE * 0.5
        
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
        
        # === EXIT ON REGIME/TREND CHANGE ===
        if in_position and position_side > 0:
            # Exit long if daily turns bearish and we're in strong trend
            if price_below_hma_1d and hma_1d_slope_down and chop_value < 40:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if daily turns bullish and we're in strong trend
            if price_above_hma_1d and hma_1d_slope_up and chop_value < 40:
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