#!/usr/bin/env python3
"""
Experiment #747: 1d Primary + 1w HTF — Connors RSI Mean Reversion + HMA Trend Filter

Hypothesis: After analyzing 746 experiments, clear patterns emerge for 1d timeframe:
1. Pure trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
2. Connors RSI mean reversion with trend filter shows Sharpe +0.923 on ETH in research
3. 1d timeframe naturally limits trades to 10-30/year (perfect for fee minimization)
4. 1w HTF HMA provides robust trend bias without overfitting

Strategy design:
1. 1w HMA(21) for primary trend bias (very slow, robust)
2. Connors RSI(3,2,100) for mean reversion entries (proven 75% win rate)
3. ADX(14) > 18 filter to avoid dead chop (looser than >25)
4. ATR(14) trailing stop 2.5x for risk management
5. Discrete signals: 0.0, ±0.25, ±0.30

Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 15 + price > 1w HMA (oversold in uptrend)
- Short: CRSI > 85 + price < 1w HMA (overbought in downtrend)

Key differences from failed experiments:
- NO complex dual-regime switching (caused 0 trades)
- NO Choppiness Index (failed in 6+ experiments)
- Loose CRSI thresholds (15/85 not 10/90) to ensure trade frequency
- 1w HTF for trend (slower = more robust than 1d/4h)
- Clear hold logic to maintain mean reversion positions

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 10-30 trades/year per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_1w_adx_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of Connors RSI - measures consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    # Calculate streak values (positive for up, negative for down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_delta = np.diff(streak)
    gain = np.where(streak_delta > 0, streak_delta, 0)
    loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + rs))
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI - current price vs lookback distribution."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period + 1:
        return pr
    
    for i in range(period, n):
        lookback = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(lookback < current)
        pr[i] = (rank / (period - 1)) * 100
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 5:
        return crsi
    
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    # All three components must be valid
    valid = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(pr)
    crsi[valid] = (rsi_3[valid] + streak_rsi[valid] + pr[valid]) / 3.0
    
    return crsi

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / (atr + 1e-10)
        minus_di = 100 * minus_di / (atr + 1e-10)
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(250, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(adx[i]):
            continue
        
        # === TREND BIAS (1w HTF HMA) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === ADX FILTER (avoid dead chop) ===
        not_dead_chop = adx[i] > 18  # Loose threshold to ensure trades
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (Mean Reversion in Uptrend) ===
        long_signal = False
        
        # Path 1: CRSI oversold (<15) + 1w bullish + not dead chop
        if crsi[i] < 15 and trend_1w_bullish and not_dead_chop:
            long_signal = True
        
        # Path 2: CRSI very oversold (<10) + above SMA50 (stronger confirmation)
        if crsi[i] < 10 and above_sma50:
            long_signal = True
        
        # Path 3: CRSI oversold (<20) + above SMA200 + 1w bullish (deep pullback in bull)
        if crsi[i] < 20 and above_sma200 and trend_1w_bullish:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (Mean Reversion in Downtrend) ===
        short_signal = False
        
        # Path 1: CRSI overbought (>85) + 1w bearish + not dead chop
        if crsi[i] > 85 and trend_1w_bearish and not_dead_chop:
            short_signal = True
        
        # Path 2: CRSI very overbought (>90) + below SMA50 (stronger confirmation)
        if crsi[i] > 90 and below_sma50:
            short_signal = True
        
        # Path 3: CRSI overbought (>80) + below SMA200 + 1w bearish (rally in bear)
        if crsi[i] > 80 and below_sma200 and trend_1w_bearish:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals (rare), go with 1w HMA trend
        if long_signal and short_signal:
            if trend_1w_bullish:
                desired_signal = BASE_SIZE
            elif trend_1w_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if mean reversion thesis intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1w HMA still bullish (trend intact)
                if trend_1w_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1w HMA still bearish (trend intact)
                if trend_1w_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses OR CRSI becomes overbought (mean reversion complete)
            if trend_1w_bearish or crsi[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses OR CRSI becomes oversold (mean reversion complete)
            if trend_1w_bullish or crsi[i] < 30:
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
                # Position flip
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