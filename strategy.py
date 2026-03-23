#!/usr/bin/env python3
"""
Experiment #781: 4h Primary + 1d HTF — Choppiness Index Regime + KAMA Trend + CRSI Mean Reversion

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. Choppiness Index (CHOP) is superior to ADX for crypto regime detection
   - CHOP > 61.8 = ranging (mean reversion works)
   - CHOP < 38.2 = trending (trend following works)
   - This is more stable than ADX hysteresis which flip-flops
2. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA
   - Efficiency Ratio adjusts smoothing constant dynamically
   - Reduces whipsaw in choppy markets, follows trends in trending markets
3. Simpler CRSI thresholds (<20 long, >80 short) generate more trades than extreme (<10, >90)
   - Current best uses too strict thresholds = not enough trades
4. 1d HTF KAMA50 for trend bias (slower, more reliable than 12h EMA)
5. ATR trailing stop at 2.5x with proper position tracking
6. Position sizing: 0.25-0.30 discrete levels to minimize fee churn

Strategy design:
1. 1d KAMA(50) for trend bias (aligned via mtf_data helper)
2. 4h Choppiness Index(14) for regime detection
3. 4h Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. 4h KAMA(21) for dynamic support/resistance
5. 4h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30
7. Target: 25-50 trades/year on 4h timeframe

Key improvements from #764:
- Replaced ADX with Choppiness Index (more stable regime detection)
- Replaced EMA with KAMA (adaptive to volatility)
- Relaxed CRSI thresholds (<20/>80 vs <15/>85) for more trades
- Simpler entry logic (less overfitting)
- Better hold logic (maintain until opposite signal or stoploss)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_kama_crsi_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 1e-10:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2  # square for smoother adaptation
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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
    """
    Connors RSI Streak component.
    Measures consecutive up/down bars.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    # Calculate streak values
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_score = streak.copy()
    
    # Calculate RSI of streak scores
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <20 = oversold, >80 = overbought.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    # Handle NaN values
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppy vs trending.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j - 1]), 
                        abs(low[j] - close[j - 1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10, fast_period=2, slow_period=50)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(chop_4h[i]):
            continue
        
        # === TREND BIAS (1d HTF KAMA50) ===
        trend_1d_bullish = close[i] > kama_1d_aligned[i]
        trend_1d_bearish = close[i] < kama_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_range = chop_4h[i] > 61.8  # ranging market
        chop_trend = chop_4h[i] < 38.2  # trending market
        
        # === CRSI SIGNALS (simplified thresholds) ===
        crsi_oversold = crsi_4h[i] < 20
        crsi_overbought = crsi_4h[i] > 80
        crsi_neutral_low = 20 <= crsi_4h[i] <= 40
        crsi_neutral_high = 60 <= crsi_4h[i] <= 80
        
        # === KAMA POSITION ===
        above_kama_4h = close[i] > kama_4h[i]
        below_kama_4h = close[i] < kama_4h[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        if chop_range:
            # Mean reversion long: CRSI oversold + below KAMA + 1d trend not bearish
            if crsi_oversold and below_kama_4h and not trend_1d_bearish:
                desired_signal = BASE_SIZE
            
            # Mean reversion short: CRSI overbought + above KAMA + 1d trend not bullish
            if crsi_overbought and above_kama_4h and not trend_1d_bullish:
                desired_signal = -BASE_SIZE
            
            # Conservative: extreme CRSI + trend alignment
            if crsi_4h[i] < 15 and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_4h[i] > 85 and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) ===
        elif chop_trend:
            # Trend pullback long: 1d bullish + CRSI neutral low + below KAMA
            if trend_1d_bullish and crsi_neutral_low and below_kama_4h:
                desired_signal = BASE_SIZE
            
            # Trend pullback short: 1d bearish + CRSI neutral high + above KAMA
            if trend_1d_bearish and crsi_neutral_high and above_kama_4h:
                desired_signal = -BASE_SIZE
            
            # Trend continuation: 1d bullish + CRSI not overbought + above KAMA
            if trend_1d_bullish and crsi_4h[i] < 70 and above_kama_4h:
                desired_signal = REDUCED_SIZE
            
            # Trend continuation: 1d bearish + CRSI not oversold + below KAMA
            if trend_1d_bearish and crsi_4h[i] > 30 and below_kama_4h:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only extreme CRSI + strong trend alignment
            if crsi_4h[i] < 15 and trend_1d_bullish and above_kama_4h:
                desired_signal = REDUCED_SIZE
            
            if crsi_4h[i] > 85 and trend_1d_bearish and below_kama_4h:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend intact and CRSI not overbought
                if trend_1d_bullish and crsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and CRSI not oversold
                if trend_1d_bearish and crsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses or CRSI overbought
            if trend_1d_bearish and crsi_4h[i] > 70:
                desired_signal = 0.0
            # Exit if price breaks below KAMA in ranging regime
            if chop_range and below_kama_4h and crsi_4h[i] > 50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses or CRSI oversold
            if trend_1d_bullish and crsi_4h[i] < 30:
                desired_signal = 0.0
            # Exit if price breaks above KAMA in ranging regime
            if chop_range and above_kama_4h and crsi_4h[i] < 50:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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