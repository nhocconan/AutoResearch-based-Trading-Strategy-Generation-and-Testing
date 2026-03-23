#!/usr/bin/env python3
"""
Experiment #393: 1d Primary + 1w HTF — Regime-Adaptive CRSI + Donchian + Vol Filter

Hypothesis: Previous 1d strategies failed due to over-filtering or wrong regime detection.
This strategy combines:
1. Weekly HMA for strong directional bias (slow, reliable trend filter)
2. Connors RSI for mean-reversion entries (proven 75% win rate on ETH)
3. Donchian(20) breakout confirmation (avoids false signals in chop)
4. Volatility filter (ATR ratio) to avoid low-vol traps
5. Choppiness Index for regime detection (trend vs mean-revert mode)

Key improvements from failed experiments:
- SINGLE HTF (1w only) — dual HTF caused over-filtering in #382
- Relaxed CRSI: <35 for long, >65 for short (not <30/>70)
- Vol filter: ATR(7)/ATR(30) > 0.8 (avoid dead markets, not too strict)
- Donchian as confirmation, not primary trigger (reduces whipsaw)
- Simple 2.5x ATR trailing stop — no complex exit logic

Target: 15-30 trades/year on 1d, Sharpe > 0.5 on ALL symbols.
Must beat #392 (Sharpe=0.119) and current best (Sharpe=0.612).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_donchian_1w_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    More responsive than EMA, less lag than SMA.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    diff = 2.0 * wma_half - wma_full
    hma = diff.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Thresholds: <35 oversold (long), >65 overbought (short)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak - consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -streak_period:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0 + 25.0 * streak[i]
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - percentile of today's return vs last pr_period bars
    returns = close_s.pct_change()
    percent_rank = np.full(n, 50.0)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i]
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[i] > window).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chopperness = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chopperness = np.nan_to_num(chopperness, nan=50.0)
    chopperness = np.clip(chopperness, 0, 100)
    
    return chopperness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volatility ratio filter (avoid dead markets)
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = atr_7 / (atr_30 + 1e-10)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 1d (target 15-30 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(vol_ratio[i]):
            continue
        
        # === HTF BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA crossover) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === VOLATILITY FILTER ===
        vol_ok = vol_ratio[i] > 0.7  # Avoid extremely low vol
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        long_bias = price_above_hma_1w  # HTF bullish
        
        # Entry trigger 1: Donchian breakout confirmation
        breakout_long = close[i] > donchian_upper[i-1]
        
        # Entry trigger 2: CRSI oversold pullback (relaxed <35)
        crsi_oversold = crsi[i] < 35.0
        
        # Entry trigger 3: HMA bullish
        momentum_long = hma_bullish
        
        # LONG ENTRY conditions (RELAXED to ensure trades)
        if long_bias and vol_ok:
            if is_trending and breakout_long and momentum_long:
                # Trend breakout long (all 3 confirm)
                desired_signal = BASE_SIZE
            elif is_choppy and crsi_oversold:
                # Range mean-reversion long
                desired_signal = BASE_SIZE
            elif momentum_long and crsi_oversold and breakout_long:
                # Pullback + breakout confluence
                desired_signal = BASE_SIZE
            elif momentum_long and crsi_oversold:
                # Simple pullback in uptrend
                desired_signal = BASE_SIZE
        
        # SHORT SETUP
        short_bias = price_below_hma_1w  # HTF bearish
        
        # Entry trigger 1: Donchian breakdown
        breakout_short = close[i] < donchian_lower[i-1]
        
        # Entry trigger 2: CRSI overbought rally (relaxed >65)
        crsi_overbought = crsi[i] > 65.0
        
        # Entry trigger 3: HMA bearish
        momentum_short = hma_bearish
        
        # SHORT ENTRY conditions (RELAXED to ensure trades)
        if short_bias and vol_ok:
            if is_trending and breakout_short and momentum_short:
                # Trend breakdown short
                desired_signal = -BASE_SIZE
            elif is_choppy and crsi_overbought:
                # Range mean-reversion short
                desired_signal = -BASE_SIZE
            elif momentum_short and crsi_overbought and breakout_short:
                # Rally + breakdown confluence
                desired_signal = -BASE_SIZE
            elif momentum_short and crsi_overbought:
                # Simple rally in downtrend
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 75:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias:
                desired_signal = BASE_SIZE
            elif position_side < 0 and short_bias:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals