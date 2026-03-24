#!/usr/bin/env python3
"""
Experiment #441: 15m Primary + 1h/4h HTF — Fisher Transform + CRSI Mean Reversion

Hypothesis: 15m has ZERO successful experiments because entry conditions were TOO STRICT.
Previous 15m failures (#429, #433, #437) all had 0 trades due to over-filtering.

New approach (LOOSE conditions to guarantee trades):
1. FISHER TRANSFORM (period=9): Proven reversal indicator for bear/range markets
   Long: Fisher crosses above -1.5 | Short: Fisher crosses below +1.5
2. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 20 | Short: CRSI > 80 (75% win rate in literature)
3. 4h HMA for trend bias: Only trade WITH HTF trend direction
4. Session filter: 00-12 UTC only (London+NY overlap) - reduces trade count
5. VERY LOOSE entries: Fisher OR CRSI trigger (not AND) to ensure trades

Why this should work on 15m:
- Fisher Transform catches reversals that EMA/RSI miss
- CRSI is proven mean-reversion indicator (Connors Research)
- 4h HTF filter prevents counter-trend trades
- Session filter keeps trades ~50-80/year (not 300+)
- Size=0.20 (conservative for 15m frequency)

Target: Sharpe>0.40, DD>-35%, trades>=60 train (15/year), trades>=10 test
Timeframe: 15m (FIRST 15m strategy with LOOSE conditions)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_crsi_4h_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals in bear/range markets better than RSI
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate median price
    median = (high + low) / 2.0
    
    # Normalize price to -1 to +1 range
    fisher_input = np.zeros(n)
    fisher_input[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest - lowest > 1e-10:
            normalized = 2.0 * (median[i] - lowest) / (highest - lowest) - 1.0
            # Clamp to avoid division issues
            normalized = max(-0.999, min(0.999, normalized))
            fisher_input[i] = normalized
    
    # Fisher transform: 0.5 * ln((1+x)/(1-x))
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(fisher_input[i]):
            x = fisher_input[i]
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    # Signal line (EMA of Fisher)
    fisher_signal = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher, fisher_signal

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with 75% win rate
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    rsi_short[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(streak_period + 1, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                # Count consecutive up days
                j = i
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                # Count consecutive down days (negative)
                j = i
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        
        # Calculate RSI of streak
        if streak > 0:
            streak_rsi[i] = 100.0 - (100.0 / (streak + 1))
        elif streak < 0:
            streak_rsi[i] = 100.0 / (abs(streak) + 1)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and not np.all(np.isnan(returns)):
            current_return = returns[-1] if len(returns) > 0 else 0
            valid_returns = returns[~np.isnan(returns)]
            if len(valid_returns) > 0:
                percent_rank[i] = np.sum(valid_returns < current_return) / len(valid_returns) * 100.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
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

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate primary (15m) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
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
        
        if np.isnan(fisher[i]) or np.isnan(crsi[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = (hour >= 0 and hour < 12)
        
        # === 4h HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h HTF CONFIRMATION ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            # Long: Fisher crosses above -1.5 (oversold reversal)
            if fisher[i-1] <= -1.5 and fisher[i] > -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 (overbought reversal)
            if fisher[i-1] >= 1.5 and fisher[i] < 1.5:
                fisher_cross_short = True
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === RSI FILTER (loose) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === ENTRY LOGIC (LOOSE - Fisher OR CRSI) ===
        desired_signal = 0.0
        
        # LONG ENTRIES (with HTF bullish bias)
        if htf_4h_bull or htf_1h_bull:
            # Fisher reversal OR CRSI oversold (either triggers)
            if fisher_cross_long or (crsi_oversold and rsi_oversold):
                if in_session:
                    desired_signal = SIZE_BASE
            # Stronger signal: both Fisher + CRSI
            elif fisher_cross_long and crsi_oversold:
                if in_session:
                    desired_signal = SIZE_STRONG
        
        # SHORT ENTRIES (with HTF bearish bias)
        elif htf_4h_bear or htf_1h_bear:
            # Fisher reversal OR CRSI overbought (either triggers)
            if fisher_cross_short or (crsi_overbought and rsi_overbought):
                if in_session:
                    desired_signal = -SIZE_BASE
            # Stronger signal: both Fisher + CRSI
            elif fisher_cross_short and crsi_overbought:
                if in_session:
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