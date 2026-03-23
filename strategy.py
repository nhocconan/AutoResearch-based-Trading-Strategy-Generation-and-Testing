#!/usr/bin/env python3
"""
Experiment #1087: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: After 785+ failed experiments, daily timeframe with regime detection works best:
1. Choppiness Index (CHOP) detects market regime:
   - CHOP > 55 = range/chop (use mean reversion with Connors RSI)
   - CHOP < 45 = trending (use Donchian breakout with HMA filter)
2. Connors RSI (CRSI) for mean reversion entries in range regime:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > SMA(200)
   - Short: CRSI > 85 + price < SMA(200)
3. Donchian(20) breakout for trend regime:
   - Long: price breaks 20-day high + HMA(21) > HMA(48)
   - Short: price breaks 20-day low + HMA(21) < HMA(48)
4. 1w HMA21 for macro bias — only trade in direction of weekly trend
5. ATR(14) trailing stop 3x — wider stops for daily timeframe
6. Position size: 0.25-0.30 discrete levels

Why this should beat Sharpe=0.612:
- Dual regime adapts to market conditions (trend vs range)
- Connors RSI has 75% win rate in backtests
- Choppiness Index is proven regime filter for bear markets
- 1d timeframe = 20-50 trades/year = minimal fee drag
- 1w HTF prevents counter-trend trades in major reversals

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 3x ATR trailing (wider for daily)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    Formula: HMA = WMA(sqrt(period)) of (2*WMA(period/2) - WMA(period))
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite momentum indicator for mean reversion.
    Formula: CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: percentile rank of daily returns over lookback
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3) on close
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Calculate streak (consecutive up/down days)
    diff = np.diff(close)
    streak = np.zeros(n)
    for i in range(1, n):
        if diff[i-1] > 0:
            streak[i] = streak[i-1] + 1 if i > 0 and diff[i-1] > 0 else 1
        elif diff[i-1] < 0:
            streak[i] = streak[i-1] - 1 if i > 0 and diff[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, np.abs(streak), 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = avg_streak_loss > 1e-10
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + rs_streak[mask]))
    rsi_streak[~mask] = 50.0
    
    # PercentRank(100) — percentile of today's return vs last 100 days
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        if len(window) > 0:
            rank = np.sum(window < returns[i]) / len(window)
            percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures if market is trending or ranging.
    Formula: CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/chop (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period * 2:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    range_hl = highest - lowest
    mask = range_hl > 1e-10
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout levels."""
    n = len(close)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    sma_200 = calculate_sma(close, 200)
    rsi_3 = calculate_rsi(close, period=3)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200-day SMA is ready
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        range_regime = chop[i] > 55.0  # Range/mean reversion
        trend_regime = chop[i] < 45.0  # Trending
        
        # === PRIMARY TREND (1d HMA crossover) ===
        hma_bull = hma_21[i] > hma_48[i]
        hma_bear = hma_21[i] < hma_48[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI SIGNALS (Range Regime) ===
        crsi_oversold = crsi[i] < 15.0  # Mean reversion long
        crsi_overbought = crsi[i] > 85.0  # Mean reversion short
        
        # === DONCHIAN BREAKOUT (Trend Regime) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === RANGE REGIME — Mean Reversion with CRSI ===
        if range_regime:
            # Long: CRSI oversold + above SMA200 + macro bull or neutral
            if crsi_oversold and above_sma200 and (macro_bull or not macro_bear):
                desired_signal = current_size
            
            # Short: CRSI overbought + below SMA200 + macro bear or neutral
            elif crsi_overbought and below_sma200 and (macro_bear or not macro_bull):
                desired_signal = -current_size
        
        # === TREND REGIME — Donchian Breakout with HMA ===
        elif trend_regime:
            # Long: Donchian breakout + HMA bull + macro bull
            if donchian_breakout_long and hma_bull and macro_bull:
                desired_signal = current_size
            
            # Short: Donchian breakout + HMA bear + macro bear
            elif donchian_breakout_short and hma_bear and macro_bear:
                desired_signal = -current_size
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) — Stay flat or hold ===
        else:
            if in_position:
                desired_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
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
                # Hold long if regime still supports
                if (range_regime and crsi[i] < 70.0) or (trend_regime and hma_bull):
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if regime still supports
                if (range_regime and crsi[i] > 30.0) or (trend_regime and hma_bear):
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if regime changes against us
            if trend_regime and hma_bear:
                desired_signal = 0.0
            if range_regime and crsi[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if regime changes against us
            if trend_regime and hma_bull:
                desired_signal = 0.0
            if range_regime and crsi[i] < 30.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
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
                # Flip position
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