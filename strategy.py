#!/usr/bin/env python3
"""
Experiment #374: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime v1

Hypothesis: Pure trend strategies fail in bear/range markets (2025 test period).
Connors RSI (CRSI) has 75% win rate for mean reversion. Choppiness Index properly
identifies regime to switch between mean revert (chop) and trend follow (trend).

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Choppiness Index regime: CHOP>61.8=range(mean revert), CHOP<38.2=trend
3. 1w HMA for major trend bias (only trade with weekly trend)
4. ATR(14) stoploss at 2.5x from entry
5. Discrete signals: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work:
- 1d timeframe = 20-50 trades/year target (low fee drag)
- CRSI catches reversals in bear market rallies
- CHOP filter avoids mean revert in strong trends
- 1w HTF ensures we trade with major trend direction
- Conservative sizing (0.25-0.30) limits drawdown in 2022 crash

Target: Sharpe>0.45, DD>-35%, trades>=20 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI Streak component for Connors RSI
    Measures consecutive up/down days"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        up_streak = 0
        down_streak = 0
        
        # Count consecutive up days ending at i
        for j in range(i, max(i - 20, 0), -1):
            if j > 0 and close[j] > close[j-1]:
                up_streak += 1
            else:
                break
        
        # Count consecutive down days ending at i
        for j in range(i, max(i - 20, 0), -1):
            if j > 0 and close[j] < close[j-1]:
                down_streak += 1
            else:
                break
        
        # Streak value: positive for up, negative for down
        if up_streak > 0:
            streak_value = up_streak
        else:
            streak_value = -down_streak
        
        # Calculate RSI of streak values (simplified: use raw streak)
        # Connors uses RSI of streak, we approximate with normalized streak
        streak_rsi[i] = 100.0 * (up_streak / (up_streak + down_streak + 1e-10)) if (up_streak + down_streak) > 0 else 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank for Connors RSI
    Where does current price change rank vs last N days?"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(period, n):
        current_change = close[i] - close[i-1] if i > 0 else 0.0
        
        count_higher = 0
        count_total = 0
        
        for j in range(i - period, i):
            if j > 0:
                past_change = close[j] - close[j-1]
                count_total += 1
                if current_change > past_change:
                    count_higher += 1
        
        if count_total > 0:
            pct_rank[i] = 100.0 * count_higher / count_total
        else:
            pct_rank[i] = 50.0
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, rank_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_1d = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION via Choppiness Index ===
        # CHOP > 61.8 = ranging (mean reversion)
        # CHOP < 38.2 = trending (trend follow)
        # Between = use previous signals
        
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF BIAS (1w) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA TREND ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 10 = extremely oversold (long signal)
        # CRSI > 90 = extremely overbought (short signal)
        crsi_oversold = crsi[i] < 15.0  # Loosened from 10 for more trades
        crsi_overbought = crsi[i] > 85.0  # Loosened from 90 for more trades
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: RANGING (mean reversion with CRSI)
        if is_ranging:
            # Long: CRSI oversold + above SMA200 + 1w bull bias
            if crsi_oversold and above_sma200 and htf_1w_bull:
                desired_signal = SIZE_BASE
            
            # Short: CRSI overbought + below SMA200 + 1w bear bias
            elif crsi_overbought and below_sma200 and htf_1w_bear:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (trend follow with HMA)
        elif is_trending:
            # Long: 1d HMA bull + 1w HMA bull + pullback to HMA
            if hma_bull and htf_1w_bull:
                # Enter on pullback (close near HMA but still above)
                pullback_long = (close[i] > hma_1d[i]) and (close[i] < hma_1d[i] * 1.02)
                if pullback_long:
                    desired_signal = SIZE_STRONG
            
            # Short: 1d HMA bear + 1w HMA bear + rally to HMA
            elif hma_bear and htf_1w_bear:
                # Enter on rally (close near HMA but still below)
                rally_short = (close[i] < hma_1d[i]) and (close[i] > hma_1d[i] * 0.98)
                if rally_short:
                    desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * atr[i]
                else:
                    stop_price = entry_price + 2.5 * atr[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals