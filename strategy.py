#!/usr/bin/env python3
"""
Experiment #1068: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Choppiness Regime + CRSI

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than HMA/EMA,
providing superior trend detection in both bull and bear markets. Combined with Choppiness Index
regime filter and Connors RSI for entry timing, this should work across BTC/ETH/SOL in all regimes.

Key innovations:
1. KAMA (Efficiency Ratio based): Adapts smoothing based on market noise - fast in trends, slow in chop
2. Choppiness Index (CHOP 14): >58 = range (mean revert), <42 = trend (trend follow)
3. Connors RSI (CRSI): More responsive than standard RSI for entry timing
4. 12h KAMA for intermediate trend, 1d KAMA for long-term bias
5. Regime-adaptive entries with LOOSE conditions to guarantee trades
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Why this should work:
- KAMA adapts to 2022 crash volatility better than fixed-period MA
- Choppiness filter avoids trend-following whipsaws in range markets
- CRSI has proven 75% win rate on mean reversion
- LOOSE entry conditions ensure 30+ trades on train, 5+ on test
- 4h captures multi-day swings (20-50 trades/year target)
- Works on BTC/ETH bear markets AND SOL bull markets

Entry conditions (LOOSE to guarantee trades):
- LONG range: CHOP>55 + CRSI<30 + price>1d_KAMA*0.97
- LONG trend: CHOP<45 + price>12h_KAMA>1d_KAMA + RSI(14)>45
- SHORT range: CHOP>55 + CRSI>70 + price<1d_KAMA*1.03
- SHORT trend: CHOP<45 + price<12h_KAMA<1d_KAMA + RSI(14)<55

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_crsi_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency/volatility
    ER = |Close - Close(n)| / Sum(|Close - Close(prev)|)
    SC = [ER * (fast SC - slow SC) + slow SC]^2
    KAMA = KAMA(prev) + SC * (Close - KAMA(prev))
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 1e-10:
                er[i] = signal / noise
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(close[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    rsi_3[:rsi_period] = np.nan
    
    # Component 2: RSI of Streak
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.divide(avg_streak_gain, avg_streak_loss, out=np.zeros_like(avg_streak_gain), where=avg_streak_loss != 0)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi[:streak_period] = np.nan
    
    # Component 3: Percent Rank
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window < close[i])
            percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine components
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=14)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama_4h = calculate_kama(close, period=14)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 55.0  # Range market
        is_trending = chop_14[i] < 45.0  # Trend market
        
        # === HTF BIAS (KAMA alignment) ===
        price_above_12h = close[i] > kama_12h_aligned[i]
        price_below_12h = close[i] < kama_12h_aligned[i]
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        kama_12h_above_1d = kama_12h_aligned[i] > kama_1d_aligned[i]
        kama_12h_below_1d = kama_12h_aligned[i] < kama_1d_aligned[i]
        
        # Strong trend alignment
        strong_bull = price_above_12h and price_above_1d and kama_12h_above_1d
        strong_bear = price_below_12h and price_below_1d and kama_12h_below_1d
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE, LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Connors RSI extremes
            # Long when CRSI oversold (LOOSE: <30 not <20)
            if crsi[i] < 30.0 and price_above_1d:
                desired_signal = SIZE_BASE
            # Short when CRSI overbought (LOOSE: >70 not >80)
            elif crsi[i] > 70.0 and price_below_1d:
                desired_signal = -SIZE_BASE
            # Stronger signals at more extreme CRSI
            elif crsi[i] < 18.0:
                desired_signal = SIZE_STRONG
            elif crsi[i] > 82.0:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - use KAMA alignment + RSI filter
            # Long in uptrend with RSI confirmation (LOOSE: RSI>45 not >50)
            if strong_bull and rsi_14[i] > 45.0 and rsi_14[i] < 80.0:
                desired_signal = SIZE_STRONG
            # Short in downtrend with RSI confirmation (LOOSE: RSI<55 not <50)
            elif strong_bear and rsi_14[i] < 55.0 and rsi_14[i] > 20.0:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals (LOOSE conditions)
            elif price_above_12h and price_above_1d and rsi_14[i] > 48.0:
                desired_signal = SIZE_BASE
            elif price_below_12h and price_below_1d and rsi_14[i] < 52.0:
                desired_signal = -SIZE_BASE
        
        # === FALLBACK: Simple KAMA crossover for more trades ===
        if desired_signal == 0.0:
            # Long if price crosses above 4h KAMA with HTF support
            if close[i] > kama_4h[i] and price_above_1d and rsi_14[i] > 50.0:
                desired_signal = SIZE_BASE
            # Short if price crosses below 4h KAMA with HTF support
            elif close[i] < kama_4h[i] and price_below_1d and rsi_14[i] < 50.0:
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