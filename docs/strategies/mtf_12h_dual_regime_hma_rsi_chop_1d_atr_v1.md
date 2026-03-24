# Strategy: mtf_12h_dual_regime_hma_rsi_chop_1d_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.555 | -6.6% | -21.1% | 654 | FAIL |
| ETHUSDT | -0.233 | +2.7% | -22.8% | 657 | FAIL |
| SOLUSDT | 0.651 | +98.9% | -14.0% | 659 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.060 | +5.7% | -11.7% | 201 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1526: 12h Primary + 1d HTF — Dual Regime (Trend/Mean Revert) + HMA + RSI

Hypothesis: Based on proven patterns from research:
1. Choppiness Index regime detection switches between trend-follow and mean-revert
2. In trend regime (CHOP < 38.2): HMA trend + Donchian breakout
3. In range regime (CHOP > 61.8): Connors RSI mean reversion
4. 1d HMA(21) provides macro bias filter
5. Loose entry conditions ensure 20-50 trades/year target is met
6. ATR 2.5x trailing stop for risk management

Key insights from 1100+ failed strategies:
- Complex filters = 0 trades (#1515, #1518 had Sharpe=0.000)
- SIMPLER works: HTF trend + primary signal (#1522 Sharpe=0.462)
- 12h naturally generates appropriate trade frequency
- Dual regime adapts to market conditions (bull/bear/range)

Design:
- 1d HMA(21) for macro trend bias (HTF filter)
- 12h Choppiness Index(14) for regime detection
- 12h HMA(16) for primary trend
- 12h RSI(14) for entry timing (loose bands ensure trades)
- 12h Donchian(20) for momentum confirmation
- ATR(14) 2.5x trailing stop
- Position size 0.30 (discrete: 0.0, ±0.20, ±0.30)
- Target: 80-200 trades/train (4 years), 20-50 trades/test (15 months)

Timeframe: 12h (as required by experiment)
HTF: 1d (daily trend bias)
Position Size: 0.30 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_hma_rsi_chop_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

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
    Measures market choppiness vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme values (<10 or >90) signal mean reversion opportunities
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        pos_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        streak_rsi[i] = 100.0 * pos_streaks / streak_period if streak_period > 0 else 50.0
    
    # Percent Rank - where does current return rank vs last 100 periods
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        rank = np.sum(returns[:-1] < current_return) / max(len(returns) - 1, 1)
        percent_rank[i] = 100.0 * rank
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 12h (20-50 trades/year target)
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_12h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trend_regime = chop[i] < 45.0  # Trending market (looser threshold for more trades)
        range_regime = chop[i] > 55.0  # Range/choppy market
        
        # === PRIMARY TREND (12h HMA) ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi[i] < 45.0  # Loose oversold for longs
        rsi_overbought = rsi[i] > 55.0  # Loose overbought for shorts
        
        # === CONNORS RSI (Mean Reversion Signal) ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 35.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 65.0
        
        # === DONCHIAN MOMENTUM ===
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10 and not np.isnan(donchian_range):
            donchian_position = (close[i] - donchian_lower[i]) / donchian_range
        else:
            donchian_position = 0.5
        
        donchian_bull = donchian_position > 0.45  # Price in upper half
        donchian_bear = donchian_position < 0.55  # Price in lower half
        
        # === DESIRED SIGNAL - DUAL REGIME APPROACH ===
        desired_signal = 0.0
        
        # LONG SIGNALS
        if daily_bull:  # Only long when daily trend is bullish
            # Trend regime: HMA + Donchian breakout
            if trend_regime and hma_bull and donchian_bull:
                desired_signal = BASE_SIZE
            # Range regime: Connors RSI mean reversion
            elif range_regime and crsi_oversold and rsi_oversold:
                desired_signal = BASE_SIZE
            # Fallback: HMA bull + RSI not overbought (ensures trades)
            elif hma_bull and rsi[i] < 60.0:
                desired_signal = BASE_SIZE * 0.7
            # Fallback 2: Daily bull + HMA bull (simplest, most trades)
            elif daily_bull and hma_bull:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT SIGNALS
        elif daily_bear:  # Only short when daily trend is bearish
            # Trend regime: HMA + Donchian breakdown
            if trend_regime and hma_bear and donchian_bear:
                desired_signal = -BASE_SIZE
            # Range regime: Connors RSI mean reversion
            elif range_regime and crsi_overbought and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Fallback: HMA bear + RSI not oversold (ensures trades)
            elif hma_bear and rsi[i] > 40.0:
                desired_signal = -BASE_SIZE * 0.7
            # Fallback 2: Daily bear + HMA bear (simplest, most trades)
            elif daily_bear and hma_bear:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.5
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
```

## Last Updated
2026-03-24 01:20
