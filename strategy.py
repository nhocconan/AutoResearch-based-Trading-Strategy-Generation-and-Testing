#!/usr/bin/env python3
"""
Experiment #575: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: After 512 failed strategies, the pattern is clear:
- Pure trend following fails in bear/range markets (2022 crash, 2025 test period)
- Pure mean reversion fails in strong trends
- SOLUTION: Use Choppiness Index (CHOP) to detect regime, then apply appropriate strategy
  - CHOP > 55 = range/messy → use Connors RSI mean reversion
  - CHOP < 45 = trending → use HMA trend following
  - CHOP 45-55 = transition → stay flat
- Add session filter (8-20 UTC) to avoid low-liquidity whipsaws
- Add 4h HMA for major trend bias (only trade with HTF direction)
- Target: 40-60 trades/year on 1h (Rule 10), position size 0.22 (conservative)

Why this might beat Sharpe=0.435:
1. Regime-adaptive logic handles both bull/bear/range markets
2. Connors RSI has 75% win rate in literature for mean reversion
3. Session filter avoids Asian session whipsaws (proven in #545 analysis)
4. 4h HTF filter prevents counter-trend losses in major moves
5. Discrete position sizing (0.0, ±0.22) minimizes fee churn

Position sizing: 0.22 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_crsi_session_4h_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range-bound market (mean reversion)
    - CHOP < 38.2 = trending market (trend following)
    - 38.2 - 61.8 = transition zone
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close - short-term momentum
    2. RSI(2) of streak - consecutive up/down days
    3. PercentRank(100) - where current return ranks vs last 100 bars
    
    Entry signals:
    - Long: CRSI < 10 (oversold)
    - Short: CRSI > 90 (overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=returns.index, dtype=float)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if not np.isnan(current):
            percent_rank.iloc[i] = (window < current).sum() / rank_period * 100.0
        else:
            percent_rank.iloc[i] = 50.0
    
    # Combine components
    crsi = (rsi_close.values + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
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

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF HMA for major trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for 1h vs 4h (Rule 10: lower TF = more trades = smaller size)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        session_ok = 8 <= hour <= 20
        
        # === 4H MAJOR TREND (primary direction filter) ===
        bull_regime_4h = close[i] > hma_4h_21_aligned[i]
        bear_regime_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPING INDEX REGIME DETECTION ===
        chop_high = chop_14[i] > 55.0  # Range/choppy market
        chop_low = chop_14[i] < 45.0   # Trending market
        # chop 45-55 = transition, stay flat or reduce size
        
        # === CONNORS RSI EXTREMES (mean reversion) ===
        crsi_oversold = crsi[i] < 15.0   # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        
        # === RSI PULLBACK (trend following) ===
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        
        # === ADX FILTER (ensure some momentum) ===
        adx_ok = adx_14[i] > 18.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # REGIME 1: CHOPPY/RANGE (CHOP > 55) → Mean Reversion with CRSI
        if chop_high and session_ok:
            # Long: CRSI oversold + 4h not strongly bearish
            if crsi_oversold and not (bear_regime_4h and hma_4h_slope_bear):
                new_signal = POSITION_SIZE
            # Short: CRSI overbought + 4h not strongly bullish
            elif crsi_overbought and not (bull_regime_4h and hma_4h_slope_bull):
                new_signal = -POSITION_SIZE
        
        # REGIME 2: TRENDING (CHOP < 45) → Trend Following with RSI pullback
        elif chop_low and session_ok and adx_ok:
            # Long: 4h bull + RSI pullback
            if bull_regime_4h and rsi_pullback_long:
                if hma_4h_slope_bull:
                    new_signal = POSITION_SIZE
                else:
                    new_signal = POSITION_SIZE * 0.7
            # Short: 4h bear + RSI pullback
            elif bear_regime_4h and rsi_pullback_short:
                if hma_4h_slope_bear:
                    new_signal = -POSITION_SIZE
                else:
                    new_signal = -POSITION_SIZE * 0.7
        
        # REGIME 3: TRANSITION (CHOP 45-55) → Stay flat or hold existing
        # new_signal stays 0.0
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        # Exit long on 4h regime flip to strong bear
        if in_position and position_side > 0:
            if bear_regime_4h and hma_4h_slope_bear:
                new_signal = 0.0
            # Exit on CRSI overbought in range regime
            elif chop_high and crsi_overbought:
                new_signal = 0.0
        
        # Exit short on 4h regime flip to strong bull
        if in_position and position_side < 0:
            if bull_regime_4h and hma_4h_slope_bull:
                new_signal = 0.0
            # Exit on CRSI oversold in range regime
            elif chop_high and crsi_oversold:
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