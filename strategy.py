#!/usr/bin/env python3
"""
Experiment #416: 12h Primary + 1d HTF — Simplified Regime-Adaptive CRSI + ADX

Hypothesis: After analyzing 400+ failed experiments, key insights:
1. #406 had Sharpe=0.011 — too many conflicting filters prevented trades
2. Need SIMPLER entry logic with fewer conditions (each filter reduces trade count)
3. CRSI thresholds too extreme (<15/>85) — relax to <25/>75 for more signals
4. Add 1d HMA SLOPE (not just price position) for better regime detection
5. Add ADX(14) for trend strength — only trend-follow when ADX>25
6. Asymmetric sizing: 0.35 with 1d trend, 0.20 against (reduces counter-trend risk)
7. Ensure >=30 trades/symbol by loosening entry: OR logic not AND

Why this might beat current best (Sharpe=0.435):
- 12h TF = lower fee drag (20-50 trades/year vs 100+ on 4h/1h)
- CRSI mean reversion works in range markets (60% of time per research)
- 1d HMA slope detects regime changes earlier than price cross
- ADX filter prevents trend-following in chop (reduces 2022-style whipsaw)
- Simpler logic = more trades = better statistical significance

Position sizing: 0.20-0.35 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_adx_regime_1d_simp_v2"
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
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100)
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = pct_change.iloc[i-rank_period:i]
        current = pct_change.iloc[i]
        if not np.isnan(current) and len(window) > 0:
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100.0
    
    crsi = (rsi_3 + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 1d HMA slope (regime detection)
    hma_1d_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_1d_21_aligned[i]) and not np.isnan(hma_1d_21_aligned[i-5]):
            hma_1d_slope[i] = (hma_1d_21_aligned[i] - hma_1d_21_aligned[i-5]) / (hma_1d_21_aligned[i-5] + 1e-10)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    choppiness = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_WITH_TREND = 0.35
    SIZE_COUNTER = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(choppiness[i]):
            continue
        
        # === 1D MAJOR TREND REGIME ===
        # Slope > 0.005 = strong bull, < -0.005 = strong bear, else neutral
        bull_regime = hma_1d_slope[i] > 0.002
        bear_regime = hma_1d_slope[i] < -0.002
        neutral_regime = not bull_regime and not bear_regime
        
        # Price vs 1d HMA confirmation
        price_above_1d = close[i] > hma_1d_21_aligned[i]
        price_below_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_14[i] > 25.0
        weak_trend = adx_14[i] < 20.0
        
        # === CHOPPINESS REGIME ===
        is_choppy = choppiness[i] > 55.0
        is_trending = choppiness[i] < 45.0
        
        # === CONNORS RSI (relaxed thresholds for more trades) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_neutral_low = crsi[i] < 40.0
        crsi_neutral_high = crsi[i] > 60.0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — SIMPLIFIED OR CONDITIONS ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY (multiple paths — any one triggers)
        long_signal = False
        
        # Path 1: Bull regime + mean reversion (choppy + CRSI oversold)
        if bull_regime and is_choppy and crsi_oversold:
            long_signal = True
        
        # Path 2: Bull regime + trend pullback (HMA bullish + CRSI < 50)
        if bull_regime and hma_bullish and crsi_neutral_low:
            long_signal = True
        
        # Path 3: Neutral regime + strong oversold (CRSI < 20)
        if neutral_regime and crsi[i] < 20.0:
            long_signal = True
        
        # Path 4: Frequency boost (no trade > 12 bars)
        if bars_since_last_trade > 12 and not in_position:
            if bull_regime and crsi[i] < 45.0:
                long_signal = True
            if neutral_regime and crsi_oversold and rsi_oversold:
                long_signal = True
        
        if long_signal:
            size = SIZE_WITH_TREND if bull_regime else SIZE_COUNTER
            new_signal = size
        
        # SHORT ENTRY (multiple paths — any one triggers)
        short_signal = False
        
        # Path 1: Bear regime + mean reversion (choppy + CRSI overbought)
        if bear_regime and is_choppy and crsi_overbought:
            short_signal = True
        
        # Path 2: Bear regime + trend pullback (HMA bearish + CRSI > 50)
        if bear_regime and hma_bearish and crsi_neutral_high:
            short_signal = True
        
        # Path 3: Neutral regime + strong overbought (CRSI > 80)
        if neutral_regime and crsi[i] > 80.0:
            short_signal = True
        
        # Path 4: Frequency boost (no trade > 12 bars)
        if bars_since_last_trade > 12 and not in_position:
            if bear_regime and crsi[i] > 55.0:
                short_signal = True
            if neutral_regime and crsi_overbought and rsi_overbought:
                short_signal = True
        
        if short_signal and new_signal == 0.0:
            size = SIZE_WITH_TREND if bear_regime else SIZE_COUNTER
            new_signal = -size
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit
        if in_position and position_side > 0 and hma_bearish and weak_trend:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish and weak_trend:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals