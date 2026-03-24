#!/usr/bin/env python3
"""
Experiment #026: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 25 failed experiments, the pattern shows higher timeframes (1d/1w) work best.
This strategy combines three proven patterns from literature:
1. Connors RSI (CRSI) - proven 75% win rate for mean reversion entries
2. Choppiness Index regime detection - switches between trend/mean-revert modes
3. 1w HMA for major trend bias - keeps us on right side of macro trend

Key innovations vs failed attempts:
- CRSI is MORE sensitive than regular RSI, generates more signals
- Dual regime: mean-revert in chop (CHOP>55), trend-follow in clean trends (CHOP<45)
- LOOSE entry thresholds to ensure >=30 trades on train, >=3 on test
- 1d timeframe targets 20-50 trades/year (low fee drag)
- Position size 0.28 (28% capital) conservative for daily swings
- ATR 2.5x trailing stoploss for risk management

Why this should beat current best (Sharpe=0.019):
- CRSI catches reversals better than standard RSI (proven on ETH +0.923)
- Choppiness filter avoids trend strategies in choppy markets (major failure mode)
- 1w HTF bias prevents counter-trend trades in strong macro moves
- Designed specifically for bear/range markets (2022 crash, 2025 bear)

Target: Sharpe>0.35, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_v1"
timeframe = "1d"
leverage = 1.0

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Larry Connors' mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    for i in range(streak_period + 1, n):
        streak_window = streak[i-streak_period:i+1]
        delta_streak = np.diff(streak_window)
        gain_streak = np.sum(np.where(delta_streak > 0, delta_streak, 0))
        loss_streak = np.sum(np.where(delta_streak < 0, -delta_streak, 0))
        
        if loss_streak < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = gain_streak / loss_streak
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100) - percentile of today's return
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1]) / close[i-rank_period:i]
        if len(returns) > 0:
            today_return = returns[-1]
            rank = np.sum(returns < today_return) / len(returns)
            percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use 45-55 as neutral zone for regime switching
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
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=34)
    sma_200 = calculate_sma(close, 200)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 1d)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d[i]) or np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = choppy/range (mean revert)
        # CHOP < 45 = trending (trend follow)
        # 45-55 = neutral (use HTF bias)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        is_neutral = not is_choppy and not is_trending
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 10 = extreme oversold (long)
        # CRSI > 90 = extreme overbought (short)
        # Use looser thresholds to ensure trades: <15 / >85
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === SMA 200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === 1d HMA TREND ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_choppy:
            # CHOPPY REGIME: Mean reversion with CRSI
            # LONG: CRSI oversold + above SMA200 + HTF not strongly bear
            if crsi_oversold and above_sma200 and not htf_bear:
                desired_signal = SIZE
            # SHORT: CRSI overbought + below SMA200 + HTF not strongly bull
            elif crsi_overbought and below_sma200 and not htf_bull:
                desired_signal = -SIZE
            # Fallback: extreme CRSI with HMA confirmation
            elif crsi[i] < 10.0 and hma_bull:
                desired_signal = SIZE * 0.7
            elif crsi[i] > 90.0 and hma_bear:
                desired_signal = -SIZE * 0.7
        
        elif is_trending:
            # TRENDING REGIME: Follow HTF trend with CRSI pullback entries
            # LONG: HTF bull + CRSI pullback (not extreme) + HMA bull
            if htf_bull and crsi[i] < 50.0 and hma_bull and above_sma200:
                desired_signal = SIZE
            # SHORT: HTF bear + CRSI pullback (not extreme) + HMA bear
            elif htf_bear and crsi[i] > 50.0 and hma_bear and below_sma200:
                desired_signal = -SIZE
        
        else:
            # NEUTRAL REGIME: Use HTF bias with CRSI extremes
            if htf_bull and crsi_oversold:
                desired_signal = SIZE
            elif htf_bear and crsi_overbought:
                desired_signal = -SIZE
            # Fallback: HMA crossover with CRSI
            elif hma_bull and crsi[i] < 40.0:
                desired_signal = SIZE * 0.5
            elif hma_bear and crsi[i] > 60.0:
                desired_signal = -SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals