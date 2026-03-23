#!/usr/bin/env python3
"""
Experiment #939: 4h Primary + 1d HTF — Connors RSI Mean Reversion + Donchian Breakout + HMA Trend

Hypothesis: After 668 failed strategies, simplifying to proven Connors RSI mean reversion
with HMA trend filter should generate consistent trades across ALL symbols (BTC/ETH/SOL).

Key insights from research:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > HMA_1d (bullish macro pullback)
   - Short: CRSI > 85 + price < HMA_1d (bearish macro rally)
   - Historical win rate: 75% on mean reversion entries
2. Donchian(20) breakout confirms trend direction
3. 1d HMA(21) for macro regime filter (simple but effective)
4. ATR(14) trailing stop at 2.5x for risk management

Why this should work:
- CRSI is proven mean reversion indicator (Larry Connors research)
- Simpler conditions = more trades (avoiding 0-trade failure)
- 1d HMA filter prevents counter-trend trades in strong trends
- 4h timeframe targets 25-40 trades/year (low fee drag)
- Works in both bull and bear markets (mean reversion always exists)

Critical improvements from #934:
- REMOVED funding rate (alignment issues, not always available)
- REMOVED choppiness index (adds complexity, minimal benefit)
- SIMPLIFIED regime logic (just HMA_1d, not 12h+1d+chop)
- LOOSENED entry thresholds (CRSI < 15/>85 not extreme values)
- DISCRETE signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

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
    """RSI Streak: consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(period, n):
        up_streak = max(0, streak[i])
        down_streak = abs(min(0, streak[i]))
        total = up_streak + down_streak + 1e-10
        streak_rsi[i] = 100 * up_streak / total
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank: where current return ranks vs last N periods."""
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        pct_rank[i] = 100 * rank / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3."""
    rsi_3 = calculate_rsi(close, rsi_period)
    streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.full(n, np.nan)
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_3[i] + streak[i] + pct_rank[i]) / 3
    
    return crsi

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
    """Donchian Channel: highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

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
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 15
        crsi_overbought = crsi_4h[i] > 85
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.005
        donchian_mid_above = close[i] > donchian_mid[i]
        donchian_mid_below = close[i] < donchian_mid[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: CRSI oversold + macro bullish (mean reversion in uptrend)
        if crsi_oversold and macro_bull:
            desired_signal = BASE_SIZE
        # Secondary: CRSI extreme oversold (strong mean reversion)
        elif crsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        # Tertiary: Donchian breakout + macro bullish (trend follow)
        elif donchian_breakout_long and macro_bull and donchian_mid_above:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Primary: CRSI overbought + macro bearish (mean reversion in downtrend)
        if crsi_overbought and macro_bear:
            if desired_signal > 0:
                desired_signal = 0.0  # Cancel long if short signal stronger
            desired_signal = -BASE_SIZE
        # Secondary: CRSI extreme overbought (strong mean reversion)
        elif crsi_extreme_overbought:
            if desired_signal > 0:
                desired_signal = 0.0
            desired_signal = -REDUCED_SIZE
        # Tertiary: Donchian breakdown + macro bearish (trend follow)
        elif donchian_breakout_short and macro_bear and donchian_mid_below:
            if desired_signal > 0:
                desired_signal = 0.0
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
                # Hold long if macro bullish and CRSI not overbought
                if macro_bull and crsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bearish and CRSI not oversold
                if macro_bear and crsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + CRSI overbought
            if macro_bear and crsi_4h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + CRSI oversold
            if macro_bull and crsi_4h[i] < 30:
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