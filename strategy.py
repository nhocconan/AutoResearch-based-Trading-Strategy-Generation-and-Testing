#!/usr/bin/env python3
"""
Experiment #676: 30m Primary + 4h/1d HTF — CRSI Mean Reversion + Choppiness Regime + HTF HMA Bias

Hypothesis: 30m timeframe is too fast for pure trend following (fee drag) but perfect for
mean reversion entries when aligned with HTF trend. Connors RSI (CRSI) has 75% win rate
in literature for mean reversion. Combined with Choppiness Index to detect range vs trend
regimes, and 4h/1d HMA for directional bias.

Key innovations:
1. CRSI(3,2,100) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — extreme readings reverse
2. Choppiness(14) > 55 = range market (favor mean reversion), < 45 = trend (reduce size)
3. 4h HMA(21) + 1d HMA(50) dual HTF bias — only long above both, short below both
4. Session filter 08-20 UTC — avoid low liquidity periods that cause false breakouts
5. Asymmetric entries: long when CRSI<20, short when CRSI>80 (loose enough for trades)
6. ATR(14) trailing stop 2.5x — protects from trend reversals

Why this should work:
- 30m CRSI extremes happen frequently (ensures trades)
- HTF bias prevents counter-trend mean reversion (2022 crash protection)
- Choppiness filter reduces whipsaw in trending markets
- Session filter avoids Asian low-liquidity fakeouts

Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-30%
Timeframe: 30m
Size: 0.20 discrete (0.0, ±0.20, ±0.30)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_htf_hma_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    mask = avg_loss > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + avg_gain[mask] / avg_loss[mask]))
    rsi[~mask & (avg_gain > 0)] = 100.0
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of CRSI — consecutive up/down days"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        # Count consecutive up days
        up_streak = 0
        down_streak = 0
        for j in range(i, max(0, i - 20), -1):
            if j == 0:
                break
            if close[j] > close[j - 1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j - 1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        # Convert streak to RSI-like value (0-100)
        max_streak = max(up_streak, down_streak, 1)
        if up_streak > 0:
            streak_rsi[i] = 100.0 * up_streak / (up_streak + 1)
        elif down_streak > 0:
            streak_rsi[i] = 100.0 * (1.0 / (down_streak + 1))
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of CRSI — where current close ranks in lookback"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period - 1, n):
        lookback = close[i - period + 1:i + 1]
        count_below = np.sum(lookback[:-1] < close[i])
        pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    rsi = calculate_rsi(close, rsi_period)
    streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + streak + pr) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures if market is choppy or trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(highest_high - lowest_low) / np.log10(tr_sum)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
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
        # Convert open_time (ms) to hour
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h + 1d HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 55.0  # Range market — favor mean reversion
        is_trend = chop[i] < 45.0  # Trending market — reduce mean reversion size
        
        # === CRSI EXTREMES (LOOSE THRESHOLDS FOR TRADES) ===
        crsi_oversold = crsi[i] < 25.0  # Long signal
        crsi_overbought = crsi[i] > 75.0  # Short signal
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO ENSURE TRADES) ===
        desired_signal = 0.0
        
        # LONG: HTF bull/neutral + CRSI oversold + (range market OR in session)
        if htf_bull and crsi_oversold:
            if is_range or in_session:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        elif htf_neutral and crsi_oversold and is_range:
            # Neutral HTF but range market — smaller mean reversion long
            desired_signal = SIZE_BASE * 0.5
        
        # SHORT: HTF bear/neutral + CRSI overbought + (range market OR in session)
        elif htf_bear and crsi_overbought:
            if is_range or in_session:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        elif htf_neutral and crsi_overbought and is_range:
            # Neutral HTF but range market — smaller mean reversion short
            desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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