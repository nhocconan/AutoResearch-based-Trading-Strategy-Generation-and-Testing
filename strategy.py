#!/usr/bin/env python3
"""
Experiment #286: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian Breakout v1

Hypothesis: Daily timeframe with weekly HTF filter provides optimal balance of signal quality
and trade frequency (20-50 trades/year). Choppiness Index cleanly separates mean-reversion
from trending regimes. Connors RSI excels at catching oversold/overbought extremes in chop.
Donchian breakouts capture trend continuation. 1w HMA provides major trend bias.

Key design choices:
1. CHOPPINESS REGIME: CHOP > 61.8 = mean revert (CRSI entries), CHOP < 38.2 = trend (Donchian)
2. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
3. LOOSENED THRESHOLDS: CRSI < 30 / > 70 (not 20/80) to ensure sufficient trades on 1d
4. 1W HMA FILTER: Only take longs when price > 1w HMA, shorts when price < 1w HMA
5. ASYMMETRIC SIZING: 0.30 when 1w aligned, 0.20 otherwise
6. ATR STOPLOSS: 2.5x ATR from entry (fixed, not trailing — more reliable)
7. NO FUNDING RATE: Removed due to loading issues in #284, focus on price action only

Target: Sharpe>0.40, DD>-40%, trades>=20 train, trades>=3 test
Position sizing: 0.20-0.30 discrete levels to minimize fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average — faster response than EMA, less lag"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index — standard Wilder's calculation"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range — Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
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
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with ~75% win rate on extremes
    """
    n = len(close)
    if n < pr_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) — short-term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak calculation — consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Streak RSI — RSI of the streak values
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        up_count = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        streak_rsi[i] = 100.0 * up_count / streak_period
    
    # Percent Rank of returns — where does today's return rank vs last 100 days
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0 and not np.isnan(returns[i]):
            count_below = np.sum(valid_window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(valid_window)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis (avoid rapid flipping)
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(chop[i]):
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 58.0  # Slightly lower to get more mean-revert signals
        trending_threshold = 42.0  # Slightly higher to confirm strong trend
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory in transition zone
        
        prev_regime = current_regime
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA TREND ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === HMA CROSSOVER (fast vs slow) ===
        hma_cross_bull = hma_1d_fast[i] > hma_1d[i]
        hma_cross_bear = hma_1d_fast[i] < hma_1d[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === CRSI VALUES (LOOSENED for 1d to ensure trades) ===
        crsi_extreme_low = False
        crsi_extreme_high = False
        if not np.isnan(crsi[i]):
            crsi_extreme_low = crsi[i] < 30.0  # Was 20, loosened for 1d
            crsi_extreme_high = crsi[i] > 70.0  # Was 80, loosened for 1d
        
        # === ADX TREND STRENGTH ===
        adx_strong = not np.isnan(adx[i]) and adx[i] > 25.0
        adx_weak = not np.isnan(adx[i]) and adx[i] < 20.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with CRSI)
        if current_regime == 2:
            # Long: oversold + above SMA200 + 1w bull bias
            if crsi_extreme_low and above_sma200:
                if htf_bull:
                    desired_signal = SIZE_STRONG
                elif hma_bull:
                    desired_signal = SIZE_BASE
            
            # Short: overbought + below SMA200 + 1w bear bias
            elif crsi_extreme_high and below_sma200:
                if htf_bear:
                    desired_signal = -SIZE_STRONG
                elif hma_bear:
                    desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (breakout with HMA + HTF confirmation)
        elif current_regime == 1:
            # Long: Donchian breakout + HMA bull + 1w bull + ADX strong
            if breakout_long and hma_bull and htf_bull and adx_strong:
                desired_signal = SIZE_STRONG
            
            # Short: Donchian breakout + HMA bear + 1w bear + ADX strong
            elif breakout_short and hma_bear and htf_bear and adx_strong:
                desired_signal = -SIZE_STRONG
            
            # Weaker trend entries (ADX weak but HMA cross)
            elif breakout_long and hma_cross_bull and htf_bull:
                desired_signal = SIZE_BASE
            elif breakout_short and hma_cross_bear and htf_bear:
                desired_signal = -SIZE_BASE
        
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
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals