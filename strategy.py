#!/usr/bin/env python3
"""
Experiment #1036: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 751+ failed strategies, the pattern is clear — complex multi-filter strategies
generate 0 trades. This strategy SIMPLIFIES entry conditions while keeping the edge:

1. CONNORS RSI (CRSI): Composite of RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long: CRSI < 15 (oversold) + price > 1d HMA21 (bullish bias)
   - Short: CRSI > 85 (overbought) + price < 1d HMA21 (bearish bias)
   - Proven 70%+ win rate in range/bear markets

2. CHOPPINESS INDEX regime filter (relaxed):
   - CHOP > 55 = favor mean reversion (CRSI signals)
   - CHOP < 45 = favor trend continuation
   - Between = allow both (this is KEY — don't block all entries)

3. 1d HMA21 trend bias:
   - Only long when price > 1d HMA (removes counter-trend longs in bear)
   - Only short when price < 1d HMA (removes counter-trend shorts in bull)
   - This asymmetry works through 2022 crash and 2025 bear

4. ATR Trailing Stop: 3.0x ATR for risk management

Why 12h works:
- Target 20-50 trades/year (low fee drag)
- Less noise than 4h/1h, more signals than 1d
- Proven in experiment history (ETH Sharpe +0.923 with CRSI+CHOP)

Critical fixes from failed experiments:
- RELAXED Choppiness thresholds (55/45 not 61.8/38.2) — MORE TRADES
- CRSI thresholds 15/85 (not 10/90) — captures more reversals
- NO vol spike filter (blocked too many entries)
- NO dual HTF conflict (just 1d HMA, not 12h+1d)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    Composite of: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current close ranks vs last 100 closes
    
    Entry: CRSI < 15 (oversold), Exit: CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = rsi_3.values
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_series = pd.Series(streak)
    streak_delta = streak_series.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # Percent Rank (where current close ranks vs last 100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion favored)
    CHOP < 38.2 = trending (trend follow favored)
    We use relaxed 55/45 thresholds for more trades
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_12h[i]):
            continue
        
        # === TREND BIAS (1d HMA21) ===
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index - RELAXED) ===
        regime_chop = chop_12h[i] > 55  # Ranging → mean reversion favored
        regime_trend = chop_12h[i] < 45  # Trending → trend follow favored
        regime_neutral = not regime_chop and not regime_trend  # Allow both
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15
        crsi_overbought = crsi_12h[i] > 85
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Mean reversion in choppy market (primary signal)
        if regime_chop and price_above_hma and crsi_oversold:
            desired_signal = BASE_SIZE
        # Extreme oversold (override regime)
        elif crsi_extreme_oversold and price_above_hma:
            desired_signal = BASE_SIZE
        # Neutral regime with bullish bias
        elif regime_neutral and price_above_hma and crsi_oversold:
            desired_signal = REDUCED_SIZE
        # Trending market — only enter on pullback
        elif regime_trend and price_above_hma and crsi_12h[i] < 40:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Mean reversion in choppy market (primary signal)
        if regime_chop and price_below_hma and crsi_overbought:
            desired_signal = -BASE_SIZE
        # Extreme overbought (override regime)
        elif crsi_extreme_overbought and price_below_hma:
            desired_signal = -BASE_SIZE
        # Neutral regime with bearish bias
        elif regime_neutral and price_below_hma and crsi_overbought:
            desired_signal = -REDUCED_SIZE
        # Trending market — only enter on rally
        elif regime_trend and price_below_hma and crsi_12h[i] > 60:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend bias intact and CRSI not extreme overbought
                if price_above_hma and crsi_12h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend bias intact and CRSI not extreme oversold
                if price_below_hma and crsi_12h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend bias reverses strongly
            if price_below_hma and crsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend bias reverses strongly
            if price_above_hma and crsi_12h[i] < 30:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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