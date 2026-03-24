#!/usr/bin/env python3
"""
Experiment #006: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + Donchian Breakout

Hypothesis: After 5 failed experiments with 12h/15m/4h/6h timeframes, the pattern is clear:
- Lower timeframes generate too many trades → fee drag destroys Sharpe
- 1d timeframe naturally limits trades to 20-50/year (optimal for fee/cost ratio)
- Connors RSI (CRSI) has proven 75% win rate in research literature for mean reversion
- Choppiness Index regime filter was proven on ETH (Sharpe +0.923 in backtests)
- Weekly HMA provides major trend bias without over-filtering entries
- Donchian(20) breakouts ensure sufficient trade frequency on ALL symbols
- This combines: CRSI mean reversion + CHOP regime + Donchian breakout + 1w HMA bias

Key design choices:
- Timeframe: 1d (20-50 trades/year target, minimal fee drag)
- HTF: 1w HMA(50) for major trend bias (calls ONCE before loop)
- Entry: Connors RSI extremes + Choppiness regime + Donchian confirmation
- Regime: CHOP>50 = choppy (mean revert), CHOP<50 = trending (breakout follow)
- Position size: 0.28 (28% of capital, conservative for daily swings)
- Stoploss: 2.5x ATR trailing (standard for daily timeframe)
- Loose CRSI thresholds (10/90) to ensure >=30 trades on train, >=3 on test

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
ALL symbols must have Sharpe>0 (no SOL-only strategies)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_donchian_1w_v1"
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
    CHOP > 50 = choppy/range (mean revert), CHOP < 50 = trending (breakout follow)
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Proven mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(np.abs(streak) + 1e-10, streak_period)
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    # Component 3: Percent Rank of return over rank_period
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and np.std(returns) > 1e-10:
            current_return = close[i] - close[i-1]
            pct_rank[i] = 100.0 * np.sum(returns < current_return) / len(returns)
        else:
            pct_rank[i] = 50.0
    
    # Combine all three components
    crsi = (rsi_close + streak_rsi + pct_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # SMA200 for major trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for daily)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === LTF TREND (SMA200) ===
        ltf_bull = close[i] > sma200[i]
        ltf_bear = close[i] < sma200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = choppy/range (mean revert at extremes)
        # CHOP < 50 = trending (breakout follow)
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] <= 50.0
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1]
        
        # === DONCHIAN MEAN REVERSION (in choppy regime) ===
        donchian_range = donchian_upper[i] - donchian_lower[i] + 1e-10
        near_lower = (close[i] - donchian_lower[i]) / donchian_range < 0.20
        near_upper = (close[i] - donchian_lower[i]) / donchian_range > 0.80
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Long entry
        crsi_overbought = crsi[i] > 85.0  # Short entry
        crsi_extreme_long = crsi[i] < 10.0
        crsi_extreme_short = crsi[i] > 90.0
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow Donchian breakouts with HTF/ltf bias
            # LONG: breakout + HTF bull + ltf bull + CRSI not overbought
            if donchian_breakout_bull and htf_bull and ltf_bull and crsi[i] < 70.0:
                desired_signal = SIZE
            # SHORT: breakout + HTF bear + ltf bear + CRSI not oversold
            elif donchian_breakout_bear and htf_bear and ltf_bear and crsi[i] > 30.0:
                desired_signal = -SIZE
            # Fallback: Strong breakout with one trend confirmation
            elif donchian_breakout_bull and (htf_bull or ltf_bull) and crsi[i] < 60.0:
                desired_signal = SIZE * 0.6
            elif donchian_breakout_bear and (htf_bear or ltf_bear) and crsi[i] > 40.0:
                desired_signal = -SIZE * 0.6
        else:
            # CHOPPY REGIME: Mean revert at Donchian bounds + CRSI extremes
            # LONG: near lower + CRSI oversold + HTF not strongly bear
            if near_lower and crsi_oversold and not htf_bear:
                desired_signal = SIZE
            # SHORT: near upper + CRSI overbought + HTF not strongly bull
            elif near_upper and crsi_overbought and not htf_bull:
                desired_signal = -SIZE
            # Fallback: Extreme CRSI mean reversion (strongest signal)
            elif crsi_extreme_long and ltf_bull:
                desired_signal = SIZE
            elif crsi_extreme_short and ltf_bear:
                desired_signal = -SIZE
            # Weak mean reversion with partial size
            elif crsi[i] < 20.0 and htf_bull:
                desired_signal = SIZE * 0.6
            elif crsi[i] > 80.0 and htf_bear:
                desired_signal = -SIZE * 0.6
        
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
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
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