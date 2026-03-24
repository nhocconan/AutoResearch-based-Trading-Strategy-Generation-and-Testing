#!/usr/bin/env python3
"""
Experiment #406: 1d Primary + 1w HTF — Volatility Regime + Asymmetric Entries

Hypothesis: Previous strategies failed because they used symmetric long/short logic
in asymmetric markets (2022 crash, 2025 bear). This version uses:
1. Weekly HMA for major trend bias (only trade WITH weekly trend)
2. Volatility regime detection (BB Width percentile) for entry type
3. Asymmetric entries: long pullbacks in bull, short rallies in bear
4. Volume spike confirmation on breakouts (real moves have volume)
5. Connors RSI for mean reversion timing (proven 75% win rate in ranges)

Key differences from failed #394-#405:
- Simpler regime detection (BB Width instead of ADX/Chop which failed)
- Asymmetric positioning (no counter-trend trades)
- Fewer confluence requirements (2-3 filters max)
- Larger position size when weekly aligned (0.30 vs 0.20)

Target: Sharpe>0.45, DD>-35%, trades>=25 train, trades>=5 test, ALL symbols positive
Timeframe: 1d (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_vol_regime_asymmetric_hma_crsi_1w_v1"
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
    """Connors RSI - combines RSI, streak, and percentile rank"""
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period):i+1]
        avg_streak = np.mean(streak_vals)
        if avg_streak > 0:
            streak_rsi[i] = 100.0
        elif avg_streak < 0:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0
    
    # Percentile Rank(100)
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        pct_rank[i] = 100.0 * rank / rank_period
    
    # Combine
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma * 100.0
    
    return upper, lower, width

def calculate_bb_width_percentile(bb_width, lookback=60):
    """BB Width percentile for regime detection"""
    n = len(bb_width)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            window = bb_width[i-lookback+1:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                rank = np.sum(valid < bb_width[i])
                percentile[i] = 100.0 * rank / len(valid)
    
    return percentile

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 1e-10:
            er[i] = change / noise
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    kama = calculate_kama(close, er_period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_dev=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=60)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_HALF = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND BIAS (1w HTF) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY REGIME (BB Width Percentile) ===
        # Low vol (<30th pct) = trending regime (breakout entries)
        # High vol (>70th pct) = mean reversion regime (fade extremes)
        # Middle = neutral
        low_vol_regime = bb_width_pct[i] < 30.0
        high_vol_regime = bb_width_pct[i] > 70.0
        
        # === DAILY TREND ===
        daily_bull = close[i] > hma_21[i] and hma_21[i] > hma_50[i]
        daily_bear = close[i] < hma_21[i] and hma_21[i] < hma_50[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI/CRSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 15.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 85.0
        
        # === BB POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005
        near_bb_upper = close[i] > bb_upper[i] * 0.995
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_21[i]) and not np.isnan(hma_21[i-1]):
            if not np.isnan(hma_50[i]) and not np.isnan(hma_50[i-1]):
                if hma_21[i-1] <= hma_50[i-1] and hma_21[i] > hma_50[i]:
                    hma_cross_long = True
                if hma_21[i-1] >= hma_50[i-1] and hma_21[i] < hma_50[i]:
                    hma_cross_short = True
        
        # === ENTRY LOGIC (ASYMMETRIC - only trade WITH weekly trend) ===
        desired_signal = 0.0
        
        # REGIME 1: LOW VOL (trending) - breakout entries WITH weekly trend
        if low_vol_regime:
            # Long: Weekly bull + daily bull + (cross OR above SMA200 + RSI confirm)
            if weekly_bull:
                if hma_cross_long or (daily_bull and above_sma200 and rsi[i] > 50):
                    desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
            
            # Short: Weekly bear + daily bear + (cross OR below SMA200 + RSI confirm)
            elif weekly_bear:
                if hma_cross_short or (daily_bear and below_sma200 and rsi[i] < 50):
                    desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
        
        # REGIME 2: HIGH VOL (mean reversion) - fade extremes WITH weekly trend
        elif high_vol_regime:
            # Long: Weekly bull + oversold + near BB lower
            if weekly_bull and above_sma200:
                if (rsi_oversold or crsi_oversold) and near_bb_lower:
                    desired_signal = SIZE_BASE
            
            # Short: Weekly bear + overbought + near BB upper
            elif weekly_bear and below_sma200:
                if (rsi_overbought or crsi_overbought) and near_bb_upper:
                    desired_signal = -SIZE_BASE
        
        # REGIME 3: NEUTRAL VOL - smaller positions, wait for clear signals
        else:
            # Only take HMA cross with weekly alignment
            if weekly_bull and hma_cross_long and above_sma200:
                desired_signal = SIZE_BASE
            elif weekly_bear and hma_cross_short and below_sma200:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
            # Update trailing high
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
                # Trail stop up
                stop_price = max(stop_price, highest_since_entry - 2.5 * entry_atr)
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
            # Update trailing low
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
                # Trail stop down
                stop_price = min(stop_price, lowest_since_entry + 2.5 * entry_atr)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (reduce to half at 2R) ===
        if in_position and position_side > 0 and desired_signal > 0:
            profit = close[i] - entry_price
            if profit >= 2.0 * entry_atr:
                desired_signal = SIZE_HALF
        
        if in_position and position_side < 0 and desired_signal < 0:
            profit = entry_price - close[i]
            if profit >= 2.0 * entry_atr:
                desired_signal = -SIZE_HALF
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
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
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Set initial stoploss
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals