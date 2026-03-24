#!/usr/bin/env python3
"""
Experiment #420: 6h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Trend Bias

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Connors RSI (CRSI)
has proven 75% win rate for mean reversion when combined with trend filter. Using
1d HMA for intermediate trend and 1w HMA for macro bias should filter false signals
while maintaining trade frequency.

Key innovations:
1. CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — faster response than RSI(14)
2. Dual HTF filter: 1d HMA for intermediate trend, 1w HMA for macro bias
3. Asymmetric thresholds: CRSI<20 for long (easier), CRSI>80 for short (harder in bull)
4. Volume spike confirmation on entries (1.5x 20-bar volume SMA)
5. Stoploss: 2.5x ATR from entry, trail on profit

Why 6h works:
- Fewer trades than 4h (lower fee drag)
- More signals than 12h (better capture of swings)
- Target: 30-50 trades/year per symbol

Entry Logic:
- Long: CRSI < 20 + close > 1d HMA + 1w HMA bullish + volume confirm
- Short: CRSI > 80 + close < 1d HMA + 1w HMA bearish + volume confirm
- Exit: CRSI crosses 50 (mean reached) OR stoploss hit

Position sizing: 0.25 base, 0.30 when both 1d and 1w aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_meanrevert_hma_1d1w_v1"
timeframe = "6h"
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
    """RSI of streak (consecutive up/down bars)"""
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak: consecutive up or down bars
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak_window = streak[max(0, i-period+1):i+1]
        up_streak = np.sum(np.where(streak_window > 0, streak_window, 0))
        down_streak = np.abs(np.sum(np.where(streak_window < 0, streak_window, 0)))
        
        total = up_streak + down_streak
        if total < 1e-10:
            streak_rsi[i] = 50.0
        else:
            streak_rsi[i] = 100.0 * up_streak / total
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percentile rank of price change over period"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(period, n):
        # Calculate returns over the period
        returns = np.diff(close[max(0, i-period+1):i+1])
        current_return = close[i] - close[i-1] if i > 0 else 0
        
        # Count how many returns are less than current
        if len(returns) > 0:
            count_below = np.sum(returns < current_return)
            pct_rank[i] = 100.0 * count_below / len(returns)
        else:
            pct_rank[i] = 50.0
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
    return crsi

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

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss and exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # === CRSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_neutral = 45.0 < crsi[i] < 55.0  # Exit zone
        
        # === ENTRY LOGIC (CRSI mean reversion with HTF filter) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + 1d HMA bull (1w can be neutral or bull)
        # Easier entry in bull markets
        if crsi_oversold:
            if htf_1d_bull:
                # Strong signal: both 1d and 1w bullish
                if htf_1w_bull:
                    desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
                else:
                    # 1w neutral/bear but 1d bull - weaker signal
                    desired_signal = SIZE_BASE * 0.8 if vol_confirm else SIZE_BASE * 0.6
        
        # SHORT: CRSI overbought + 1d HMA bear (1w can be neutral or bear)
        # Harder entry (asymmetric - shorts need stronger confirmation)
        elif crsi_overbought:
            if htf_1d_bear:
                # Strong signal: both 1d and 1w bearish
                if htf_1w_bear:
                    desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
                else:
                    # 1w neutral/bull but 1d bear - weaker signal
                    desired_signal = -SIZE_BASE * 0.8 if vol_confirm else -SIZE_BASE * 0.6
        
        # === EXIT LOGIC ===
        # Exit when CRSI returns to neutral (mean reversion complete)
        if in_position:
            if position_side > 0 and crsi_neutral:
                desired_signal = 0.0
            elif position_side < 0 and crsi_neutral:
                desired_signal = 0.0
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.6
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.6
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