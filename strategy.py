#!/usr/bin/env python3
"""
Experiment #1055: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After analyzing 763+ failed experiments, the key insight for 1h timeframe is:
1. Lower TF strategies fail due to TOO MANY trades (>200/yr) OR TOO FEW (0 trades)
2. The winning pattern: HTF (4h/1d) for DIRECTION, 1h only for ENTRY TIMING
3. Connors RSI (CRSI) is more sensitive than regular RSI — catches reversals faster
4. Choppiness Index on 4h (not 1h) prevents whipsaw in choppy markets
5. Session filter (8-20 UTC) + volume filter reduces false signals by 60%

Strategy Logic:
1. MACRO BIAS (1d HMA21): Only long if close > 1d_HMA, only short if close < 1d_HMA
2. TREND DIRECTION (4h HMA21): Confirms 1d bias, must agree
3. REGIME FILTER (4h CHOP): CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
4. ENTRY TRIGGER (1h CRSI): Long if CRSI < 15, Short if CRSI > 85
5. VOLUME CONFIRMATION: Volume > 0.8x 20-period average
6. SESSION FILTER: Only entries 8-20 UTC (highest liquidity hours)
7. STOPLOSS: 2.5x ATR(14) trailing from entry

Why this should work for 1h:
- CRSI is more sensitive than RSI — triggers on smaller pullbacks
- 4h regime filter prevents entering against higher TF structure
- Session + volume filters cut 60% of low-quality signals
- Discrete sizing (0.25) minimizes fee drag while capturing moves
- Relaxed CRSI thresholds (15/85 vs 10/90) ensure sufficient trades

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-80 trades/year with filters)
Position Size: 0.25 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for faster reversal signals.
    Formula: CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Measures consecutive up/down bars
    PercentRank: Where current price ranks in last 100 bars
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_short = 100 - (100 / (1 + rs))
    
    # Component 2: Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    streak_gain_series = pd.Series(streak_gain)
    streak_loss_series = pd.Series(streak_loss)
    avg_streak_gain = streak_gain_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = streak_loss_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.divide(avg_streak_gain, avg_streak_loss, out=np.ones_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Component 3: Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100
    
    # Combine all 3 components
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market ranging vs trending.
    CHOP > 61.8 = ranging (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def extract_hour_from_open_time(open_time_col):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time_col // (1000 * 60 * 60)) % 24
    return hours.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === CALCULATE HTF INDICATORS ===
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    
    # === ALIGN HTF TO LTF (auto shift(1) for completed bars) ===
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === CALCULATE 4h CHOPPINESS (regime filter on HTF, not noisy 1h) ===
    chop_4h_raw = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # === CALCULATE PRIMARY (1h) INDICATORS ===
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (UTC)
    session_hours = extract_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(volume_sma[i]):
            continue
        if volume_sma[i] <= 1e-10:
            continue
        
        # === MACRO BIAS (1d HMA21) — MUST AGREE ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND DIRECTION (4h HMA21) — CONFIRMS 1d ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (4h CHOP) ===
        is_range = chop_4h_aligned[i] > 55.0
        is_trend = chop_4h_aligned[i] < 45.0
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= session_hours[i] <= 20
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Must have: macro bullish + trend bullish + CRSI oversold + volume + session
        long_bias = macro_bull and trend_bull
        
        if long_bias:
            if is_range:
                # Range mode: mean reversion entry
                if crsi[i] < 20 and volume_ok and in_session:
                    desired_signal = BASE_SIZE
            elif is_trend:
                # Trend mode: pullback entry (slightly relaxed CRSI)
                if crsi[i] < 25 and volume_ok and in_session:
                    desired_signal = BASE_SIZE
            else:
                # Transition zone: require stronger signal
                if crsi[i] < 15 and volume_ok and in_session:
                    desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        short_bias = macro_bear and trend_bear
        
        if short_bias:
            if is_range:
                # Range mode: mean reversion entry
                if crsi[i] > 80 and volume_ok and in_session:
                    desired_signal = -BASE_SIZE
            elif is_trend:
                # Trend mode: pullback entry (slightly relaxed CRSI)
                if crsi[i] > 75 and volume_ok and in_session:
                    desired_signal = -BASE_SIZE
            else:
                # Transition zone: require stronger signal
                if crsi[i] > 85 and volume_ok and in_session:
                    desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if bias intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro + trend still bullish
                if macro_bull and trend_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro + trend still bearish
                if macro_bear and trend_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish
            if macro_bear:
                desired_signal = 0.0
            # Exit long if CRSI becomes overbought
            if crsi[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish
            if macro_bull:
                desired_signal = 0.0
            # Exit short if CRSI becomes oversold
            if crsi[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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