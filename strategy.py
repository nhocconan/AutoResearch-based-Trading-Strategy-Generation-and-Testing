#!/usr/bin/env python3
"""
Experiment #834: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + CRSI + Choppiness Regime

Hypothesis: After 571+ failed strategies, the key insight is that 4h timeframe needs
ADAPTIVE trend detection (KAMA) combined with regime-aware entries (Choppiness + CRSI).
KAMA adapts to volatility — flat in choppy markets, responsive in trends. This should
work better than static EMA/HMA across all market regimes (bull/bear/range).

Strategy design:
1. 4h Primary timeframe (target 25-45 trades/year)
2. 12h KAMA(20) for adaptive trend bias (not entry trigger)
3. 1d HMA(21) for long-term secular trend filter
4. 4h Choppiness Index(14) for regime detection (CHOP>55=range, CHOP<45=trend)
5. 4h Connors RSI (CRSI) for mean reversion entries (better than plain RSI)
6. 4h ADX(14) for trend strength confirmation (ADX>25 = real trend)
7. 4h ATR(14) for trailing stop (2.5x) and volatility scaling
8. Dual regime: mean revert when CHOP>55, trend follow when CHOP<45+ADX>25
9. KAMA slope for trend direction (more adaptive than HMA crossover)

Why KAMA + CRSI:
- KAMA Efficiency Ratio adapts smoothing based on volatility
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — 75% win rate on reversals
- Choppiness Index clearly separates regime (proven in experiment #823 notes)
- ADX filter prevents trend entries in weak trends (reduces whipsaw)
- 4h timeframe balances trade frequency with signal quality

Key changes from failed 4h strategies:
- KAMA instead of HMA/EMA (adaptive to volatility)
- CRSI instead of RSI(14) (better mean reversion signal)
- ADX>25 filter for trend entries (avoid weak trends)
- CHOP thresholds: 55/45 (clearer regime separation)
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 15 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 25-45 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_chop_adx_regime_12h1d_atr_v2"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=20, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts smoothing based on volatility.
    KAMA is flat in choppy markets, responsive in trending markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines RSI(3), RSI-Streak(2), and PercentRank(100).
    Range 0-100. <10 = oversold, >90 = overbought. 75% win rate on reversals.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        pct_below = np.sum(window[:-1] < window[-1]) / (rank_period - 1)
        percent_rank[i] = pct_below * 100
    
    # CRSI = average of three components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 3:
        return adx
    
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    for i in range(1, n):
        prev_close = close[i-1]
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            dm_plus[i] = max(high[i] - high[i-1], 0)
        else:
            dm_plus[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            dm_minus[i] = max(low[i-1] - low[i], 0)
        else:
            dm_minus[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    di_plus = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    di_minus = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus_vals = 100 * di_plus / (atr + 1e-10)
        di_minus_vals = 100 * di_minus / (atr + 1e-10)
        dx = 100 * np.abs(di_plus_vals - di_minus_vals) / (di_plus_vals + di_minus_vals + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=20)
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 12h KAMA for medium-term trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=20)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate and align 1d HMA for long-term secular trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND BIAS (12h HTF KAMA20) ===
        trend_12h_bullish = close[i] > kama_12h_aligned[i]
        trend_12h_bearish = close[i] < kama_12h_aligned[i]
        
        # === KAMA SLOPE (4h) — Adaptive trend direction ===
        kama_slope_bullish = kama_4h[i] > kama_4h[i-5] if not np.isnan(kama_4h[i-5]) else False
        kama_slope_bearish = kama_4h[i] < kama_4h[i-5] if not np.isnan(kama_4h[i-5]) else False
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === TREND STRENGTH (4h ADX) ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        # === CRSI SIGNALS (Connors RSI) ===
        crsi_oversold = crsi_4h[i] < 15
        crsi_overbought = crsi_4h[i] > 85
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        crsi_recovery = crsi_4h[i] > 20 and crsi_4h[i-1] < 20 if not np.isnan(crsi_4h[i-1]) else False
        crsi_weakening = crsi_4h[i] < 80 and crsi_4h[i-1] > 80 if not np.isnan(crsi_4h[i-1]) else False
        
        # === KAMA POSITION ===
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI extreme oversold + price below KAMA (pullback in range)
            if crsi_extreme_oversold and price_below_kama:
                desired_signal = BASE_SIZE
            
            # Short: CRSI extreme overbought + price above KAMA (rally in range)
            if crsi_extreme_overbought and price_above_kama:
                desired_signal = -BASE_SIZE
            
            # CRSI recovery from oversold + any HTF trend alignment
            if crsi_oversold and crsi_recovery and (trend_1d_bullish or trend_12h_bullish):
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # CRSI weakening from overbought + any HTF trend alignment
            if crsi_overbought and crsi_weakening and (trend_1d_bearish or trend_12h_bearish):
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Fallback: extreme CRSI alone (guarantees trades on all symbols)
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45 + ADX > 25) — Trend Following ===
        elif trending_regime and strong_trend:
            # Long: Bullish trend alignment + CRSI pullback (not oversold)
            if trend_1d_bullish and trend_12h_bullish and kama_slope_bullish:
                if 20 <= crsi_4h[i] <= 50 and price_above_kama:
                    desired_signal = BASE_SIZE
                elif crsi_recovery and price_above_kama:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: Bearish trend alignment + CRSI rally (not overbought)
            if trend_1d_bearish and trend_12h_bearish and kama_slope_bearish:
                if 50 <= crsi_4h[i] <= 80 and price_below_kama:
                    desired_signal = -BASE_SIZE
                elif crsi_weakening and price_below_kama:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes + HTF trend alignment
            if crsi_extreme_oversold and (trend_1d_bullish or trend_12h_bullish):
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (trend_1d_bearish or trend_12h_bearish):
                desired_signal = -REDUCED_SIZE
            
            # KAMA crossover + CRSI confluence
            if price_above_kama and kama_slope_bullish and crsi_4h[i] < 40:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if price_below_kama and kama_slope_bearish and crsi_4h[i] > 60:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if (trend_1d_bullish or trend_12h_bullish) and crsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1d_bearish or trend_12h_bearish) and crsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + CRSI overbought
            if trend_1d_bearish and trend_12h_bearish and crsi_4h[i] > 85:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought in ranging regime
            if ranging_regime and crsi_4h[i] > 90:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + CRSI oversold
            if trend_1d_bullish and trend_12h_bullish and crsi_4h[i] < 15:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold in ranging regime
            if ranging_regime and crsi_4h[i] < 10:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
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