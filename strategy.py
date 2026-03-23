#!/usr/bin/env python3
"""
Experiment #656: 12h Primary + 1d HTF — HMA Trend + Connors RSI + ATR Stop

Hypothesis: 12h timeframe balances trade frequency (20-50/year) with signal quality.
Connors RSI (CRSI) has documented 75% win rate for mean reversion entries in 
bear/range markets. HMA provides smoother trend filter than EMA. 1d HTF HMA 
gives macro bias without over-filtering.

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 15 (oversold), Short when CRSI > 85 (overbought)
2. HMA(21) on 12h for trend direction — smoother than EMA, less lag
3. 1d HMA(21) for macro bias — only trade with weekly trend
4. ATR(14) trailing stop at 2.5x for risk management
5. Looser CRSI thresholds (15/85 vs 10/90) to ensure adequate trade frequency

Why this should beat Sharpe=0.612:
- CRSI proven edge in 2022 crash and 2025 bear market
- 12h TF = fewer false signals than 4h, more trades than 1d
- HMA reduces whipsaw vs EMA crossover strategies (all failed)
- 1d HTF filter prevents counter-trend trades in strong macro moves
- Conservative sizing (0.30) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 20 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_atr_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi[period:] = 100.0 - (100.0 / (1.0 + rs))[period:]
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    Long signal: CRSI < 15 (oversold)
    Short signal: CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period:i+1]
        pos_count = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = 100.0 * pos_count / streak_period
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank of returns
    returns = np.diff(close) / close[:-1]
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period:i]
        current_return = returns[i-1] if i > 0 else 0.0
        rank = np.sum(window_returns < current_return)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend detection."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h = calculate_hma(close, period=21)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]):
            continue
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        trend_bullish = close[i] > hma_12h[i]
        trend_bearish = close[i] < hma_12h[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15.0
        crsi_overbought = crsi_12h[i] > 85.0
        
        # Moderate CRSI levels for trend continuation
        crsi_moderate_low = crsi_12h[i] < 40.0
        crsi_moderate_high = crsi_12h[i] > 60.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: CRSI oversold + HTF not bearish (mean reversion)
        if crsi_oversold and not htf_bearish:
            desired_signal = SIZE_LONG
        # Secondary: Trend bullish + CRSI moderate (trend pullback)
        elif trend_bullish and htf_bullish and crsi_moderate_low:
            desired_signal = SIZE_LONG
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: CRSI overbought + HTF not bullish (mean reversion)
        if crsi_overbought and not htf_bullish:
            desired_signal = -SIZE_SHORT
        # Secondary: Trend bearish + HTF bearish + CRSI moderate (trend pullback)
        elif trend_bearish and htf_bearish and crsi_moderate_high:
            desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR CRSI not extremely overbought
                if (htf_bullish or trend_bullish) and crsi_12h[i] < 80.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR CRSI not extremely oversold
                if (htf_bearish or trend_bearish) and crsi_12h[i] > 20.0:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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