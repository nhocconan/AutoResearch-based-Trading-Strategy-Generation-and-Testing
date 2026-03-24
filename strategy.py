#!/usr/bin/env python3
"""
Experiment #050: 1h Primary + 4h/1d HTF — cRSI + Choppiness Regime + HMA Trend

Hypothesis: After 49 failed experiments, the pattern for 1h is clear:
- Pure trend following fails on BTC/ETH in bear/range markets
- Pure mean reversion fails on SOL (strong trends)
- SOLUTION: Dual-regime with Choppiness Index + Connors RSI for entries
- cRSI is MORE RESPONSIVE than standard RSI (uses RSI(3) + RSI_Streak + PercentRank)
- 4h HMA provides major trend bias without being too restrictive
- Session filter (08-20 UTC) reduces noise trades during low-volume hours
- This combines: cRSI extremes (75% win rate) + Choppiness regime + HTF HMA

Key design choices:
- Timeframe: 1h (30-60 trades/year target)
- HTF: 4h HMA(21) for trend bias, 1d HMA(50) for major regime
- Entry: cRSI extremes + Choppiness regime + HTF alignment
- Regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend (follow HTF)
- Position size: 0.25 (25% of capital, conservative for 1h)
- Stoploss: 2.5x ATR trailing
- LOOSE cRSI thresholds (15/85 not 10/90) to ensure >=30 trades on train

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_session_v1"
timeframe = "1h"
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
    Connors RSI (CRSI) - Larry Connors
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    More responsive than standard RSI for mean reversion entries.
    Long when CRSI < 10-15, Short when CRSI > 85-90
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2-period RSI on up/down streaks)
    streak = np.zeros(n)
    streak[:] = np.nan
    current_streak = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            current_streak = max(1, current_streak + 1)
        elif close[i] < close[i-1]:
            current_streak = min(-1, current_streak - 1)
        else:
            current_streak = 0
        streak[i] = current_streak
    
    # Convert streak to RSI-like (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        if np.isnan(streak[i]):
            continue
        # Simple conversion: positive streak = bullish, negative = bearish
        avg_streak = np.mean(streak[i-streak_period+1:i+1])
        streak_rsi[i] = 50.0 + avg_streak * 10.0  # Scale to roughly 0-100
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank (100-period)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and not np.all(np.isnan(returns)):
            current_return = returns[-1] if not np.isnan(returns[-1]) else 0
            count_below = np.sum(returns[:-1] < current_return)
            percent_rank[i] = 100.0 * count_below / (len(returns) - 1) if len(returns) > 1 else 50.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
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

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Only trade during major market hours to reduce noise
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === HTF BIAS (4h HMA + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTFs agree
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy (mean revert)
        # CHOP < 38.2 = trending (follow trend)
        # 38.2 - 61.8 = neutral (use HTF bias)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        is_neutral = (chop[i] >= 38.2) and (chop[i] <= 61.8)
        
        # === cRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # LOOSE threshold for more trades
        crsi_overbought = crsi[i] > 85.0  # LOOSE threshold for more trades
        crsi_extreme_long = crsi[i] < 25.0
        crsi_extreme_short = crsi[i] > 75.0
        
        # === 1h HMA TREND ===
        hma_1h_bull = close[i] > hma_1h[i]
        hma_1h_bear = close[i] < hma_1h[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow HTF direction with cRSI pullback entries
            # LONG: HTF bull + cRSI pullback (not oversold, just dipping)
            if htf_strong_bull and crsi_extreme_long and hma_1h_bull and in_session:
                desired_signal = SIZE
            elif htf_4h_bull and crsi[i] < 40.0 and hma_1h_bull and in_session:
                desired_signal = SIZE * 0.7
            # SHORT: HTF bear + cRSI pullback
            elif htf_strong_bear and crsi_extreme_short and hma_1h_bear and in_session:
                desired_signal = -SIZE
            elif htf_4h_bear and crsi[i] > 60.0 and hma_1h_bear and in_session:
                desired_signal = -SIZE * 0.7
        
        elif is_choppy:
            # RANGE REGIME: Mean revert at cRSI extremes
            # LONG: cRSI oversold + HTF not strongly bear
            if crsi_oversold and not htf_strong_bear and in_session:
                desired_signal = SIZE
            elif crsi[i] < 20.0 and hma_1h_bull and in_session:
                desired_signal = SIZE * 0.7
            # SHORT: cRSI overbought + HTF not strongly bull
            elif crsi_overbought and not htf_strong_bull and in_session:
                desired_signal = -SIZE
            elif crsi[i] > 80.0 and hma_1h_bear and in_session:
                desired_signal = -SIZE * 0.7
        
        else:
            # NEUTRAL REGIME: Use HTF bias with cRSI confirmation
            # LONG: HTF bull + cRSI not overbought
            if htf_strong_bull and crsi[i] < 70.0 and hma_1h_bull and in_session:
                desired_signal = SIZE * 0.7
            # SHORT: HTF bear + cRSI not oversold
            elif htf_strong_bear and crsi[i] > 30.0 and hma_1h_bear and in_session:
                desired_signal = -SIZE * 0.7
            # Extreme cRSI always triggers (regime-independent)
            elif crsi_oversold and in_session:
                desired_signal = SIZE * 0.5
            elif crsi_overbought and in_session:
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