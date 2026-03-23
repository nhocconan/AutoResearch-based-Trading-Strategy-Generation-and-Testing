#!/usr/bin/env python3
"""
Experiment #947: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 664 failed strategies, daily timeframe with proper regime detection
should work across ALL symbols. 1d reduces noise, 1w HMA provides macro bias.

Key insights from research:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long when CRSI < 10, Short when CRSI > 90. 75% win rate in backtests.
2. Choppiness Index: CHOP(14) > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 1w HMA(21) for macro regime filter — only long if price > 1w HMA, vice versa
4. ATR(14) trailing stoploss at 2.5x for risk management
5. Multiple entry triggers to ensure trades happen (CRSI + BB + RSI)

Why 1d timeframe:
- Target 20-50 trades/year (minimal fee drag)
- Less noise than lower TF, clearer regime signals
- Works in both bull and bear markets
- Proven patterns: CRSI+CHOP for ETH, Donchian+HMA for SOL

Critical for trades:
- RELAXED CRSI thresholds (< 15 / > 85 not < 10 / > 90)
- Multiple confluence paths (CRSI OR BB OR RSI extremes)
- Hold logic maintains position through pullbacks
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_bb_atr_v1"
timeframe = "1d"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of daily returns over lookback
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    direction = np.zeros(n)  # 1 = up, -1 = down, 0 = neutral
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            direction[i] = 1
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            direction[i] = -1
            if direction[i-1] == -1:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            direction[i] = 0
            streak[i] = 0
    
    # Convert streak to RSI-like value (absolute streak, normalized)
    abs_streak = np.abs(streak)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if np.isnan(abs_streak[i]):
            continue
        # Simple transformation: map streak to 0-100 scale
        streak_rsi[i] = min(100, abs_streak[i] * 25)
    
    # Percent Rank of daily returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current_return = returns[i-1] if i > 0 else 0
            rank = np.sum(valid <= current_return) / len(valid)
            percent_rank[i] = rank * 100
    
    # Combine CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3
    
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    bandwidth = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower, bandwidth
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        bandwidth[i] = (upper[i] - lower[i]) / middle[i] if middle[i] > 0 else 0
    
    return middle, upper, lower, bandwidth

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    rsi_3 = calculate_rsi(close, period=3)
    atr_1d = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower, bb_bw = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(bb_mid[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(donch_upper[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 1e-10 else 0.5
        bb_lower_break = close[i] < bb_lower[i]
        bb_upper_break = close[i] > bb_upper[i]
        bb_extreme_low = bb_position < 0.15
        bb_extreme_high = bb_position > 0.85
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_1d[i] < 35
        rsi_overbought = rsi_1d[i] > 65
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        
        # === CRSI SIGNALS (relaxed for more trades) ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i]
        donch_breakout_short = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + BB lower
            if crsi_oversold and bb_lower_break:
                desired_signal = HIGH_CONV_SIZE
            # Long: CRSI extreme oversold alone (ensures trades)
            elif crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            # Long: BB extreme low + RSI oversold
            elif bb_extreme_low and rsi_oversold:
                desired_signal = BASE_SIZE
            # Long: RSI extreme oversold + macro support
            elif rsi_extreme_oversold and macro_bull:
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + BB upper
            if crsi_overbought and bb_upper_break:
                desired_signal = -HIGH_CONV_SIZE
            # Short: CRSI extreme overbought alone
            elif crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            # Short: BB extreme high + RSI overbought
            elif bb_extreme_high and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Short: RSI extreme overbought + macro resistance
            elif rsi_extreme_overbought and macro_bear:
                desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish macro + pullback to BB lower
            if macro_bull:
                if bb_lower_break and rsi_oversold:
                    desired_signal = HIGH_CONV_SIZE
                elif crsi_oversold:
                    desired_signal = BASE_SIZE
                # Donchian breakout long
                elif donch_breakout_long:
                    desired_signal = BASE_SIZE
            
            # Short: Bearish macro + rally to BB upper
            if macro_bear:
                if bb_upper_break and rsi_overbought:
                    desired_signal = -HIGH_CONV_SIZE
                elif crsi_overbought:
                    desired_signal = -BASE_SIZE
                # Donchian breakout short
                elif donch_breakout_short:
                    desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes with macro confluence
            if crsi_extreme_oversold and macro_bull:
                desired_signal = HIGH_CONV_SIZE
            elif crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            
            if crsi_extreme_overbought and macro_bear:
                desired_signal = -HIGH_CONV_SIZE
            elif crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            
            # Secondary: BB mean reversion
            if bb_extreme_low and desired_signal == 0:
                desired_signal = BASE_SIZE
            if bb_extreme_high and desired_signal == 0:
                desired_signal = -BASE_SIZE
        
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
                # Hold long if macro intact and CRSI not overbought
                if macro_bull and crsi[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro intact and CRSI not oversold
                if macro_bear and crsi[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + CRSI overbought
            if macro_bear and crsi[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + CRSI oversold
            if macro_bull and crsi[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = HIGH_CONV_SIZE if desired_signal >= HIGH_CONV_SIZE else BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -HIGH_CONV_SIZE if desired_signal <= -HIGH_CONV_SIZE else -BASE_SIZE
        
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