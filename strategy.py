#!/usr/bin/env python3
"""
Experiment #940: 6h Primary + 1d/1w HTF — Regime-Adaptive CHOP + CRSI + HMA

Hypothesis: 6h timeframe sits between 4h and 12h, offering better signal quality
than 4h with more trades than 12h. Using Choppiness Index to detect regime
(range vs trend) allows adaptive strategy: mean reversion in ranges (CHOP>61.8),
trend following in trends (CHOP<38.2). Connors RSI (CRSI) provides proven
mean-reversion edge with 75% win rate. Weekly HMA gives major trend bias,
daily HMA gives intermediate confirmation.

Key innovations:
1. CHOP(14) regime detection: >61.8 = range (mean revert), <38.2 = trend
2. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. 1w HMA(21) for major trend bias (only trade with weekly trend)
4. 1d HMA(21) for intermediate confirmation
5. Asymmetric entries: different thresholds for long vs short
6. ATR(14) 2.5x trailing stop for risk management
7. LOOSE entry conditions to guarantee ≥10 trades/train, ≥3/test

Entry conditions (LOOSE):
- RANGE regime (CHOP>61.8): CRSI<15 long, CRSI>85 short (mean reversion)
- TREND regime (CHOP<38.2): HMA crossover + HTF bias (trend follow)
- Weekly HMA bias: only long if price>1w_HMA, only short if price<1w_HMA
- Daily HMA confirm: adds conviction but not required

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_chop_crsi_hma_1d1w_v1"
timeframe = "6h"
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
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
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
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak: consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        if avg_streak_loss[i] > 1e-10:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_streak[i] = 100.0
    
    # PercentRank: percentile of today's return vs last rank_period days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * log10(sum(ATR, n) / (Highest High - Lowest Low)) / log10(n)
    
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if highest_high[i] > lowest_low[i] and atr_sum[i] > 0:
            chop[i] = 100.0 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_6h_16 = calculate_hma(close, period=16)
    hma_6h_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    crsi_3_2_100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_chop(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(hma_6h_16[i]) or np.isnan(hma_6h_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi_3_2_100[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA - major trend) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === HTF CONFIRM (1d HMA - intermediate trend) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (CHOP) ===
        regime_range = chop_14[i] > 61.8  # choppy/range market
        regime_trend = chop_14[i] < 38.2  # trending market
        # Neutral zone: 38.2 <= CHOP <= 61.8
        
        # === 6h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_6h_16[i-1]) and not np.isnan(hma_6h_48[i-1]):
            hma_crossover_long = (hma_6h_16[i-1] <= hma_6h_48[i-1]) and (hma_6h_16[i] > hma_6h_48[i])
            hma_crossover_short = (hma_6h_16[i-1] >= hma_6h_48[i-1]) and (hma_6h_16[i] < hma_6h_48[i])
        
        # === 6h HMA TREND ===
        hma_6h_bull = hma_6h_16[i] > hma_6h_48[i]
        hma_6h_bear = hma_6h_16[i] < hma_6h_48[i]
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi_3_2_100[i] < 15.0
        crsi_overbought = crsi_3_2_100[i] > 85.0
        crsi_very_oversold = crsi_3_2_100[i] < 10.0
        crsi_very_overbought = crsi_3_2_100[i] > 90.0
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE, LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with CRSI
        if regime_range:
            # Long: CRSI oversold + weekly bullish bias (loose)
            if crsi_oversold and htf_1w_bull:
                desired_signal = SIZE_BASE
            elif crsi_very_oversold:  # Very oversold = strong signal regardless of HTF
                desired_signal = SIZE_STRONG
            
            # Short: CRSI overbought + weekly bearish bias (loose)
            if crsi_overbought and htf_1w_bear:
                desired_signal = -SIZE_BASE
            elif crsi_very_overbought:  # Very overbought = strong signal regardless of HTF
                desired_signal = -SIZE_STRONG
        
        # TREND REGIME: Trend following with HMA
        elif regime_trend:
            # Long: HMA crossover up + weekly bullish
            if hma_crossover_long and htf_1w_bull:
                desired_signal = SIZE_STRONG
            # Long: HMA bull + daily bull (trend continuation)
            elif hma_6h_bull and htf_1d_bull and htf_1w_bull:
                desired_signal = SIZE_BASE
            
            # Short: HMA crossover down + weekly bearish
            if hma_crossover_short and htf_1w_bear:
                desired_signal = -SIZE_STRONG
            # Short: HMA bear + daily bear (trend continuation)
            elif hma_6h_bear and htf_1d_bear and htf_1w_bear:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Mixed signals, require stronger confirmation
        else:
            # Only take very strong CRSI signals
            if crsi_very_oversold and htf_1w_bull:
                desired_signal = SIZE_BASE
            if crsi_very_overbought and htf_1w_bear:
                desired_signal = -SIZE_BASE
            # Or HMA crossover with both HTF confirm
            if hma_crossover_long and htf_1d_bull and htf_1w_bull:
                desired_signal = SIZE_BASE
            if hma_crossover_short and htf_1d_bear and htf_1w_bear:
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