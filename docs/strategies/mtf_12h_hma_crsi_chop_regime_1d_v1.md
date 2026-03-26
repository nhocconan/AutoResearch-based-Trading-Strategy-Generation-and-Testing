# Strategy: mtf_12h_hma_crsi_chop_regime_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.180 | +29.2% | -12.6% | 553 | PASS |
| ETHUSDT | -0.339 | -4.0% | -26.1% | 544 | FAIL |
| SOLUSDT | 0.425 | +62.3% | -27.0% | 536 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.017 | +4.6% | -14.0% | 187 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #852: 12h Primary + 1d HTF — HMA Trend + Connors RSI + Regime Adaptive

Hypothesis: 12h timeframe with daily HTF bias provides optimal trade frequency
(20-50 trades/year) with high signal quality. Hull Moving Average reduces lag
vs EMA/KAMA. Connors RSI provides proven mean-reversion edge. Choppiness Index
enables regime-adaptive logic: trend-follow in trends, mean-revert in chop.

Key innovations:
1. 1d HMA(21) for HTF trend bias - smoother than KAMA, less lag than EMA
2. 12h HMA(16/48) dual crossover for local trend confirmation
3. Connors RSI(3,2,100) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Choppiness Index(14) regime switch: >50 = range (mean revert), <50 = trend
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- TREND REGIME (CHOP<50): LONG = 1d HMA bull + 12h HMA crossover up
- TREND REGIME (CHOP<50): SHORT = 1d HMA bear + 12h HMA crossover down
- RANGE REGIME (CHOP>50): LONG = 1d HMA bull + CRSI<25
- RANGE REGIME (CHOP>50): SHORT = 1d HMA bear + CRSI>75

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_chop_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentile of today's change vs last 100 days
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] > 1e-10:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_streak[i] = 100.0
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        changes = np.diff(close[i - rank_period:i + 1])
        current_change = changes[-1]
        count_below = np.sum(changes[:-1] < current_change)
        percent_rank[i] = count_below / (rank_period - 1) * 100.0
    
    # CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50 as threshold for simplicity
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
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_12h_16[i-1]) and not np.isnan(hma_12h_48[i-1]):
            # Fast HMA crossing above/below Slow HMA
            hma_crossover_long = (hma_12h_16[i-1] <= hma_12h_48[i-1]) and (hma_12h_16[i] > hma_12h_48[i])
            hma_crossover_short = (hma_12h_16[i-1] >= hma_12h_48[i-1]) and (hma_12h_16[i] < hma_12h_48[i])
        
        # === HMA TREND ===
        hma_12h_bull = hma_12h_16[i] > hma_12h_48[i]
        hma_12h_bear = hma_12h_16[i] < hma_12h_48[i]
        
        # === CRSI CONDITIONS ===
        crsi_oversold = crsi[i] < 25.0  # Mean reversion long
        crsi_overbought = crsi[i] > 75.0  # Mean reversion short
        crsi_neutral_long = crsi[i] < 40.0  # Loose long
        crsi_neutral_short = crsi[i] > 60.0  # Loose short
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0  # Trend regime
        chop_ranging = chop_14[i] >= 50.0  # Range regime
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if htf_1d_bull:
            # Bullish HTF bias
            if chop_trending:
                # Trend regime: use HMA crossover
                if hma_crossover_long or hma_12h_bull:
                    if hma_crossover_long:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            else:
                # Range regime: use CRSI mean reversion
                if crsi_oversold or crsi_neutral_long:
                    if crsi_oversold:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
        
        elif htf_1d_bear:
            # Bearish HTF bias
            if chop_trending:
                # Trend regime: use HMA crossover
                if hma_crossover_short or hma_12h_bear:
                    if hma_crossover_short:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            else:
                # Range regime: use CRSI mean reversion
                if crsi_overbought or crsi_neutral_short:
                    if crsi_overbought:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
```

## Last Updated
2026-03-24 20:50
