#!/usr/bin/env python3
"""
Experiment #114: 1d Primary + 1w HTF — Connors RSI + Choppiness + Weekly HMA

Hypothesis: After analyzing 113 failed experiments, the winning pattern for 1d timeframe is:
- Connors RSI (CRSI) for precise entry timing at extremes (proven 75% win rate)
- Choppiness Index to detect regime (range vs trend)
- Weekly HMA for major trend bias (avoid counter-trend trades)
- LOOSE entry thresholds to ensure >=30 trades on train, >=3 on test
- This combines: CRSI (ETH Sharpe +0.923) + CHOP regime + HTF trend filter

Key design choices:
- Timeframe: 1d (20-50 trades/year, minimal fee drag)
- HTF: 1w HMA(21) for major trend bias
- Entry: CRSI<15 (long) or CRSI>85 (short) + regime filter
- Regime: CHOP>55 = range (mean revert), CHOP<55 = trend (follow breakout)
- Position size: 0.28 (28% of capital, conservative for daily swings)
- Stoploss: 3.0x ATR trailing (wider for daily timeframe)
- CRITICAL: Loose CRSI thresholds (15/85 not 10/90) to ensure trades generate

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_v2"
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

def calculate_rsi_streak(close, period=2):
    """
    Connors RSI Streak Component
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(5, n):
        # Count consecutive up/down days
        streak = 0
        if i >= 1:
            if close[i] > close[i-1]:
                streak = 1
                j = i - 1
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                streak = -1
                j = i - 1
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        
        # Convert streak to RSI-like value (0-100)
        # Long streak up = high value, long streak down = low value
        if streak >= 0:
            streak_value = min(100.0, streak * 10.0)
        else:
            streak_value = max(0.0, 100.0 + streak * 10.0)
        
        # Simple average of last 'period' streak values
        if i >= period:
            streak_values = []
            for k in range(i-period+1, i+1):
                if k >= 5:
                    s = 0
                    if k >= 1:
                        if close[k] > close[k-1]:
                            s = 1
                            j = k - 1
                            while j > 0 and close[j] > close[j-1]:
                                s += 1
                                j -= 1
                        elif close[k] < close[k-1]:
                            s = -1
                            j = k - 1
                            while j > 0 and close[j] < close[j-1]:
                                s -= 1
                                j -= 1
                    if s >= 0:
                        streak_values.append(min(100.0, s * 10.0))
                    else:
                        streak_values.append(max(0.0, 100.0 + s * 10.0))
            
            if len(streak_values) > 0:
                streak_rsi[i] = np.mean(streak_values)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank - Connors RSI component
    Measures current price position relative to last N days
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window < current)
        pr[i] = 100.0 * count_below / period
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.zeros(n)
    sma[:] = np.nan
    for i in range(period-1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
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
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    sma200 = calculate_sma(close, 200)
    sma50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for daily)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 250 bars for SMA200
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === LONG-TERM TREND (SMA200) ===
        long_term_bull = close[i] > sma200[i]
        long_term_bear = close[i] < sma200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (mean revert)
        # CHOP < 55 = trending (trend follow)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds for trades) ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean reversion at CRSI extremes
            # LONG: CRSI oversold + HTF not strongly bear + SMA200 support
            if crsi_oversold and not htf_bear:
                desired_signal = SIZE
            # SHORT: CRSI overbought + HTF not strongly bull + SMA200 resistance
            elif crsi_overbought and not htf_bull:
                desired_signal = -SIZE
            # Fallback: Extreme CRSI with SMA50 confirmation
            elif crsi[i] < 10.0 and close[i] > sma50[i]:
                desired_signal = SIZE * 0.7
            elif crsi[i] > 90.0 and close[i] < sma50[i]:
                desired_signal = -SIZE * 0.7
        else:
            # TREND REGIME: Follow HTF trend on CRSI pullbacks
            # LONG: HTF bull + CRSI pullback (not extreme) + SMA200 support
            if htf_bull and crsi[i] < 40.0 and long_term_bull:
                desired_signal = SIZE
            # SHORT: HTF bear + CRSI bounce (not extreme) + SMA200 resistance
            elif htf_bear and crsi[i] > 60.0 and long_term_bear:
                desired_signal = -SIZE
            # Fallback: Strong HTF trend entry
            elif htf_bull and crsi[i] < 50.0:
                desired_signal = SIZE * 0.7
            elif htf_bear and crsi[i] > 50.0:
                desired_signal = -SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for daily) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
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