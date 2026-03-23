#!/usr/bin/env python3
"""
Experiment #1053: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: After analyzing 762+ failed strategies, the key insight is that 1d timeframe
with weekly macro filter provides the best risk/reward for crypto perpetuals. This strategy
combines THREE proven edges:

1. CHOPPINESS INDEX (CHOP) REGIME DETECTION on 1d:
   - CHOP(14) > 61.8 = RANGING → use MEAN REVERSION (Connors RSI)
   - CHOP(14) < 38.2 = TRENDING → use TREND FOLLOWING (Donchian breakout)
   - This meta-filter adapts to market conditions automatically

2. CONNORS RSI (CRSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > 1w_HMA21
   - Short: CRSI > 85 + price < 1w_HMA21
   - Proven 75% win rate in range markets (ETH Sharpe +0.923 in research)

3. DONCHIAN BREAKOUT for trend entries:
   - Long: price breaks Donchian(20) high + price > 1w_HMA21
   - Short: price breaks Donchian(20) low + price < 1w_HMA21
   - Proven on SOL (Sharpe +0.782 in research)

4. 1w HMA21 MACRO FILTER:
   - Only long when close > 1w_HMA21 (bullish weekly bias)
   - Only short when close < 1w_HMA21 (bearish weekly bias)
   - Prevents counter-trend trades in strong weekly trends

5. ATR TRAILING STOP: 3.0x ATR(14) from entry
   - Signal→0 when stop hit (mandatory risk management)

6. RELAXED THRESHOLDS for sufficient trades:
   - CRSI: <20 / >80 (not <10 / >90)
   - CHOP: 55-65 transition zone
   - Donchian: 20-period (not 55)

Why this should work on 1d:
- 1d naturally produces 20-50 trades/year (low fee drag)
- Weekly HMA provides strong macro filter without being too restrictive
- Dual-mode adapts to bull/bear/range conditions
- Connors RSI proven on ETH, Donchian proven on SOL, regime filter helps BTC

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Long when CRSI < 10-15, Short when CRSI > 85-90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
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
        streak_window = streak[i-streak_period+1:i+1]
        avg_streak = np.mean(np.where(streak_window > 0, streak_window, 0))
        avg_streak_loss = np.mean(np.where(streak_window < 0, -streak_window, 0))
        if avg_streak_loss > 0:
            streak_rsi[i] = 100 - (100 / (1 + avg_streak / avg_streak_loss))
        else:
            streak_rsi[i] = 100.0 if avg_streak > 0 else 0.0
    
    # Component 3: Percent Rank of daily returns over 100 days
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        percentile = np.sum(window < current) / len(window)
        percent_rank[i] = percentile * 100
    
    # Combine components
    valid_mask = (~np.isnan(rsi_short)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === MACRO TREND (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION with Connors RSI ===
        if is_range:
            # Long: CRSI oversold + macro bullish bias
            if crsi[i] < 20 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + macro bearish bias
            elif crsi[i] > 80 and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker signals
            elif crsi[i] < 15 and macro_bull:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 85 and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: Donchian Breakout ===
        elif is_trend:
            # Long: price breaks Donchian high + macro bullish
            if close[i] > donchian_upper[i-1] and macro_bull:
                desired_signal = BASE_SIZE
            # Short: price breaks Donchian low + macro bearish
            elif close[i] < donchian_lower[i-1] and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker trend signals (price near breakout)
            elif close[i] > donchian_upper[i-1] * 0.98 and macro_bull:
                desired_signal = REDUCED_SIZE
            elif close[i] < donchian_lower[i-1] * 1.02 and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TRANSITION ZONE: Use simpler logic ===
        else:
            # In transition, use CRSI extremes only with macro filter
            if crsi[i] < 15 and macro_bull:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 85 and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x) ===
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if macro_bull or (is_range and crsi[i] < 70):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if macro_bear or (is_range and crsi[i] > 30):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if macro_bear and crsi[i] > 70:
                desired_signal = 0.0
            if is_trend and close[i] < donchian_lower[i-1]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull and crsi[i] < 30:
                desired_signal = 0.0
            if is_trend and close[i] > donchian_upper[i-1]:
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals