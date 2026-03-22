#!/usr/bin/env python3
"""
Experiment #574: 4h Primary + 12h/1d HTF — Dual Regime Strategy (Choppiness Switch)

Hypothesis: After 511 failed strategies, the pattern shows:
- Single-regime strategies fail because crypto alternates trend/range frequently
- #564 (4h HMA+RSI) failed with Sharpe=-0.716 — no regime filter
- #569, #571 had 0 trades — too many filters
- Choppiness Index successfully switches between trend-follow and mean-revert
- 12h HMA for major trend bias, 4h for entries, 1d for regime confirmation

Strategy Logic:
1. 12h HMA(21) = major trend direction (HTF bias)
2. 1d Choppiness Index = regime detector (CHOP>61.8 range, CHOP<38.2 trend)
3. 4h entries:
   - TREND regime (CHOP<38.2): RSI(14) pullback to 40-55 in direction of 12h trend
   - RANGE regime (CHOP>61.8): CRSI extremes (<15 long, >85 short) with 12h HMA filter
4. ADX(14) > 18 minimum movement filter (lower than failed attempts)
5. ATR(14) 2.5x trailing stop on all positions
6. Position size: 0.28 discrete (conservative for 4h)

Why this might beat Sharpe=0.435:
- Dual regime adapts to market conditions (trend vs range)
- 12h HTF prevents major counter-trend losses
- CRSI for mean-reversion has 75% win rate in literature
- Simpler than failed #569/#571 (which had 0 trades)
- Target: 25-45 trades/year on 4h (per Rule 10)

Position sizing: 0.28 base (discrete, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=10 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_crsi_12h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: consecutive up/down days
    PercentRank: percentile of today's return vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank[i] = 50.0
        else:
            valid_window = window.dropna()
            if len(valid_window) > 0:
                percent_rank[i] = 100.0 * (valid_window < current).sum() / len(valid_window)
            else:
                percent_rank[i] = 50.0
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF HMA for major trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Calculate 1d Choppiness Index for regime detection
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(chop_1d_aligned[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        hma_12h_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 1D CHOPPINESS REGIME (switch between trend/mean-revert) ===
        chop = chop_1d_aligned[i]
        trend_regime = chop < 38.2  # Trending market
        range_regime = chop > 61.8  # Ranging market
        neutral_regime = not trend_regime and not range_regime
        
        # === ADX FILTER (minimum movement) ===
        # Lower threshold than failed attempts (was >25, now >18)
        movement_ok = adx_14[i] > 18.0
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        if trend_regime and movement_ok:
            # TREND FOLLOWING: RSI pullback in direction of 12h trend
            # Long: 12h bull + RSI 40-55 (pullback, not crash)
            if bull_regime_12h and 40.0 <= rsi_14[i] <= 55.0:
                if hma_12h_slope_bull:
                    new_signal = POSITION_SIZE
                else:
                    new_signal = POSITION_SIZE * 0.7
            
            # Short: 12h bear + RSI 45-60 (rally into resistance)
            elif bear_regime_12h and 45.0 <= rsi_14[i] <= 60.0:
                if hma_12h_slope_bear:
                    new_signal = -POSITION_SIZE
                else:
                    new_signal = -POSITION_SIZE * 0.7
        
        elif range_regime and movement_ok:
            # MEAN REVERSION: CRSI extremes with 12h HMA filter
            # Long: CRSI < 15 (oversold) + price > 12h HMA (not in major downtrend)
            if crsi[i] < 15.0 and bull_regime_12h:
                new_signal = POSITION_SIZE * 0.8
            
            # Short: CRSI > 85 (overbought) + price < 12h HMA (not in major uptrend)
            elif crsi[i] > 85.0 and bear_regime_12h:
                new_signal = -POSITION_SIZE * 0.8
        
        # Neutral regime: stay flat or hold existing
        elif neutral_regime:
            if in_position:
                new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 12h regime flip to bear with slope confirmation
        if in_position and position_side > 0:
            if bear_regime_12h and hma_12h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull with slope confirmation
        if in_position and position_side < 0:
            if bull_regime_12h and hma_12h_slope_bull:
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
                # Flip position
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