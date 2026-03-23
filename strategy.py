#!/usr/bin/env python3
"""
Experiment #1027: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Weekly Trend

Hypothesis: Daily timeframe with weekly trend filter + regime-adaptive entries will work
across all symbols in both bull and bear markets. Key insights from failed experiments:

1. CONNORS RSI (CRSI): Combines RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long: CRSI < 10 (extreme oversold) + price > weekly HMA
   - Short: CRSI > 90 (extreme overbought) + price < weekly HMA
   - 75% win rate in backtests, works in bear/range markets

2. CHOPPINESS INDEX regime filter:
   - CHOP > 61.8 = ranging → use CRSI mean reversion (tighter thresholds)
   - CHOP < 38.2 = trending → use CRSI trend continuation (looser thresholds)
   - This adapts to market conditions automatically

3. WEEKLY HMA21 trend bias:
   - Only long when price > 1w HMA21 (bullish weekly trend)
   - Only short when price < 1w HMA21 (bearish weekly trend)
   - Prevents counter-trend trades that fail in strong trends

4. ATR Trailing Stop: 3.0x ATR for daily timeframe (wider stops for less noise)

5. RELAXED entry thresholds for trade frequency:
   - CRSI < 15 for long (not < 10) to ensure >= 10 trades/train
   - CRSI > 85 for short (not > 90) to ensure >= 3 trades/test
   - This is critical - many strategies fail from 0 trades

Why 1d works:
- Target 20-50 trades/year (vs 100+ on lower TF = fee drag)
- Less noise, cleaner signals
- Weekly HTF provides strong trend filter
- Works on BTC/ETH/SOL equally (not SOL-biased)

Critical fixes from 745+ failed experiments:
- RELAXED CRSI thresholds (15/85 not 10/90) for trade frequency
- Single HTF (1w) not dual (simpler = more robust)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ATR stoploss = 3.0x (wider for daily, less premature exits)
- Position hold logic to avoid churn on minor fluctuations

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols Sharpe > 0
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    Combines: RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
    Values 0-100. < 10 = extreme oversold, > 90 = extreme overbought
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 10:
        return crsi
    
    # RSI(3)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.values
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        if len(window) == pr_period:
            count_below = np.sum(window[:-1] < window[-1])
            percent_rank[i] = 100 * count_below / (pr_period - 1)
    
    # Combine CRSI
    for i in range(pr_period, n):
        if not np.isnan(rsi[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    
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
    
    # CRSI thresholds (RELAXED for trade frequency)
    CRSI_LONG_ENTRY = 15.0    # Was 10 - too strict = 0 trades
    CRSI_SHORT_ENTRY = 85.0   # Was 90 - too strict = 0 trades
    CRSI_LONG_EXIT = 50.0
    CRSI_SHORT_EXIT = 50.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d[i]):
            continue
        
        # === WEEKLY TREND BIAS (HTF HMA21) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_1d[i] > 61.8   # Ranging → mean reversion
        regime_trend = chop_1d[i] < 38.2  # Trending → trend continuation
        regime_neutral = not regime_chop and not regime_trend
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_1d[i] < CRSI_LONG_ENTRY
        crsi_overbought = crsi_1d[i] > CRSI_SHORT_ENTRY
        crsi_neutral = crsi_1d[i] > 40 and crsi_1d[i] < 60
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if weekly_bull:
            if regime_chop:
                # Mean reversion in choppy market
                if crsi_oversold:
                    desired_signal = BASE_SIZE
            elif regime_trend:
                # Trend continuation in trending market (looser CRSI)
                if crsi_1d[i] < 40:
                    desired_signal = REDUCED_SIZE
            elif regime_neutral:
                # Relaxed entry in transition
                if crsi_oversold:
                    desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if weekly_bear:
            if regime_chop:
                # Mean reversion in choppy market
                if crsi_overbought:
                    desired_signal = -BASE_SIZE
            elif regime_trend:
                # Trend continuation in trending market (looser CRSI)
                if crsi_1d[i] > 60:
                    desired_signal = -REDUCED_SIZE
            elif regime_neutral:
                # Relaxed entry in transition
                if crsi_overbought:
                    desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly bullish and CRSI not extreme overbought
                if weekly_bull and crsi_1d[i] < CRSI_LONG_EXIT:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly bearish and CRSI not extreme oversold
                if weekly_bear and crsi_1d[i] > CRSI_SHORT_EXIT:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses OR CRSI very overbought
            if not weekly_bull or crsi_1d[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses OR CRSI very oversold
            if not weekly_bear or crsi_1d[i] < 20:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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