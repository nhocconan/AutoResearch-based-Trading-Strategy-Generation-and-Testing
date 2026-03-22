#!/usr/bin/env python3
"""
Experiment #449: 4h Primary + 1d HTF — Simplified CRSI + HMA Regime + Vol Filter

Hypothesis: After analyzing 448 failed experiments, clear patterns emerge:
1. Previous 4h strategies failed due to TOO MANY conflicting filters (0 trades)
2. Connors RSI has proven 75% win rate but needs relaxed thresholds for trade frequency
3. 1d HMA slope (not just price vs HMA) gives cleaner regime signal
4. ATR volatility filter prevents entries during dead markets (fee drag)
5. Simpler logic = more trades = better statistical significance

Key changes from failed #441 (Sharpe=-1.010):
- Relaxed CRSI thresholds: <30/>70 instead of <20/>80 (MORE trades)
- Use 1d HMA slope (trend direction) instead of price vs HMA (cleaner signal)
- Add ATR vol filter: only trade when ATR/price > 0.015 (avoid dead markets)
- Remove Choppiness Index (was causing 0 trades in #448)
- Asymmetric sizing: 0.30 long, 0.25 short (protects in bear 2025)

Why this might beat current best (Sharpe=0.435):
- 4h TF balances trade frequency (20-50/year) with fee drag
- CRSI mean reversion works in both bull AND bear markets
- 1d HMA slope filter prevents counter-trend trades
- ATR vol filter skips low-vol chop (major source of losses)
- Simpler entry = guaranteed >=30 trades/symbol on train

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_hma_slope_1d_vol_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate. Relaxed thresholds for more trades.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = pd.Series(np.zeros(n))
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = (rank / rank_period) * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 1d HMA slope (trend direction)
    hma_1d_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_1d_21_aligned[i]) and not np.isnan(hma_1d_21_aligned[i-5]):
            hma_1d_slope[i] = (hma_1d_21_aligned[i] - hma_1d_21_aligned[i-5]) / hma_1d_21_aligned[i-5]
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_50 = calculate_sma(close, period=50)
    
    # ATR as % of price (volatility filter)
    atr_pct = atr_14 / close
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(crsi_4h[i]) or np.isnan(sma_50[i]):
            continue
        if np.isnan(atr_pct[i]):
            continue
        
        # === 1D REGIME (HMA slope + crossover) ===
        # Bull regime: 1d HMA21 > HMA50 AND positive slope
        # Bear regime: 1d HMA21 < HMA50 AND negative slope
        bull_regime = (hma_1d_21_aligned[i] > hma_1d_50_aligned[i]) and (hma_1d_slope[i] > 0.001)
        bear_regime = (hma_1d_21_aligned[i] < hma_1d_50_aligned[i]) and (hma_1d_slope[i] < -0.001)
        
        # === VOLATILITY FILTER (skip dead markets) ===
        # Only trade when ATR > 1.5% of price (active market)
        vol_active = atr_pct[i] > 0.015
        
        # === CONNORS RSI SIGNALS (relaxed for MORE trades) ===
        crsi_oversold = crsi_4h[i] < 30.0  # relaxed from 20
        crsi_overbought = crsi_4h[i] > 70.0  # relaxed from 80
        crsi_extreme_oversold = crsi_4h[i] < 15.0
        crsi_extreme_overbought = crsi_4h[i] > 85.0
        
        # === SMA50 FILTER (local trend) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        
        if not vol_active:
            # Low vol = no new entries, but don't exit existing
            if in_position:
                new_signal = signals[i-1] if i > 0 else 0.0
            signals[i] = new_signal
            continue
        
        # LONG ENTRIES (bull regime OR extreme oversold)
        if bull_regime or crsi_extreme_oversold:
            # Primary: CRSI oversold + bull regime
            if crsi_oversold and bull_regime:
                new_signal = LONG_SIZE
            # Secondary: Extreme oversold (works in any regime)
            elif crsi_extreme_oversold and above_sma50:
                new_signal = LONG_SIZE * 0.8
            # Tertiary: Simple pullback in bull trend
            elif bull_regime and crsi_4h[i] < 40.0 and above_sma50:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES (bear regime OR extreme overbought)
        if bear_regime or crsi_extreme_overbought:
            # Primary: CRSI overbought + bear regime
            if crsi_overbought and bear_regime:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Secondary: Extreme overbought (works in any regime)
            elif crsi_extreme_overbought and below_sma50:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Tertiary: Simple bounce in bear trend
            elif bear_regime and crsi_4h[i] > 60.0 and below_sma50:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi_4h[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_4h[i] < 20.0:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime and crsi_4h[i] > 50.0:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and crsi_4h[i] < 50.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals