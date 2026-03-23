#!/usr/bin/env python3
"""
Experiment #177: 1d Primary + 1w HTF — Donchian Breakout + Connors RSI + Choppiness Regime

Hypothesis: Previous 1d strategies failed due to overly complex regime logic and strict
entry conditions (0 trades). This strategy combines:
1. Donchian(20) breakout for trend entry (proven on SOL)
2. Connors RSI for pullback entries in trends (75% win rate in research)
3. Choppiness Index to switch between trend-follow and mean-revert modes
4. 1w HMA for macro bias filter (only trade with weekly trend)
5. Loose entry thresholds to ensure ≥30 trades/symbol on train

KEY IMPROVEMENTS:
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Entry when CRSI < 20 (long) or > 80 (short) WITH trend confirmation
- Choppiness > 55 = range mode (mean revert at Donchian bounds)
- Choppiness < 45 = trend mode (breakout entries)
- Position size: 0.30 full, 0.20 partial (discrete levels)
- ATR trailing stop at 2.5x for risk management

TARGET: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_crsi_chop_1w_v1"
timeframe = "1d"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    returns = close_s.pct_change().values
    returns[0] = 0
    
    # Calculate streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if returns[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * min(streak_abs[i], streak_period) / streak_period
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 - min(streak_abs[i], streak_period) / streak_period)
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: Percentile Rank of price change
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) > 0:
            percent_rank[i] = 100.0 * np.sum(window <= returns[i]) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    
    # Calculate 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === REGIME DETECTION ===
        chop_value = chop_14[i]
        is_trending = chop_value < 50.0  # More lenient for more trades
        is_ranging = chop_value > 50.0
        
        # === HTF MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        hma_21_above_50 = hma_21[i] > hma_50[i] if not np.isnan(hma_50[i]) else False
        hma_21_below_50 = hma_21[i] < hma_50[i] if not np.isnan(hma_50[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25.0  # Looser threshold for more trades
        crsi_overbought = crsi[i] > 75.0
        
        # CRSI turning up from oversold
        crsi_turning_up = crsi[i] > crsi[i-1] and crsi[i-1] < 30.0 if i > 0 else False
        crsi_turning_down = crsi[i] < crsi[i-1] and crsi[i-1] > 70.0 if i > 0 else False
        
        # === VOLUME FILTER (lenient) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries - multiple patterns
        # Pattern 1: Trend mode + Donchian breakout + weekly bias
        if is_trending and breakout_long and price_above_hma_1w and volume_ok:
            new_signal = POSITION_SIZE_FULL
        
        # Pattern 2: Range mode + CRSI oversold + price above weekly HMA
        elif is_ranging and crsi_oversold and price_above_hma_1w:
            new_signal = POSITION_SIZE_HALF
        
        # Pattern 3: CRSI turning up from oversold + trend confirmation
        elif crsi_turning_up and price_above_hma_21 and price_above_hma_1w:
            new_signal = POSITION_SIZE_HALF
        
        # Pattern 4: Simple pullback in uptrend (HMA21 > HMA50 + CRSI low)
        elif hma_21_above_50 and crsi_oversold and price_above_hma_1w:
            new_signal = POSITION_SIZE_HALF
        
        # SHORT entries - multiple patterns
        # Pattern 1: Trend mode + Donchian breakdown + weekly bias
        if is_trending and breakout_short and price_below_hma_1w and volume_ok:
            new_signal = -POSITION_SIZE_FULL
        
        # Pattern 2: Range mode + CRSI overbought + price below weekly HMA
        elif is_ranging and crsi_overbought and price_below_hma_1w:
            new_signal = -POSITION_SIZE_HALF
        
        # Pattern 3: CRSI turning down from overbought + trend confirmation
        elif crsi_turning_down and price_below_hma_21 and price_below_hma_1w:
            new_signal = -POSITION_SIZE_HALF
        
        # Pattern 4: Simple pullback in downtrend (HMA21 < HMA50 + CRSI high)
        elif hma_21_below_50 and crsi_overbought and price_below_hma_1w:
            new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1d HMA21
                if price_above_hma_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1d HMA21
                if price_below_hma_21:
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
        # Exit long if price crosses below 1d HMA21 significantly
        if in_position and position_side > 0 and price_below_hma_21:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d HMA21 significantly
        if in_position and position_side < 0 and price_above_hma_21:
            new_signal = 0.0
        
        # Exit if macro bias flips strongly against position
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        if in_position and position_side < 0 and price_above_hma_1w:
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