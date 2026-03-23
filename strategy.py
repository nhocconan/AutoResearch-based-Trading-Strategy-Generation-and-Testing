#!/usr/bin/env python3
"""
Experiment #877: 1d Primary + 1w HTF — Simplified Regime + CRSI + Donchian

Hypothesis: After 600+ failed strategies, the key issue is OVERCOMPLEXITY.
Strategies with 3+ HTF filters and complex regime logic generate 0 trades.

Key insights from research:
1. 1d timeframe should target 20-50 trades/year (1-2 per month)
2. Use SINGLE HTF filter (1w HMA) not multiple (1d+1w conflicts)
3. Choppiness Index for regime: CHOP>55=range (mean revert), CHOP<45=trend (breakout)
4. Connors RSI extremes (15/85 not 10/90) ensure trades trigger
5. Donchian(20) breakout for trend entries with HTF alignment
6. ATR(14) trailing stop 2.5x for risk management
7. DISCRETE signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Why this should work on 1d:
- SIMPLIFIED logic: 2 regimes (range/trend) not 3
- RELAXED CRSI thresholds (15/85) guarantee entry triggers
- SINGLE HTF filter (1w HMA) avoids conflicting signals
- Target: 25-40 trades over 4 years = ~7-10 trades/year per symbol
- ALL symbols must have positive Sharpe (no SOL-only bias)

Critical fixes from failed experiments:
- Remove conflicting HTF filters (1d+1w caused 0 trades)
- Relax CRSI from 10/90 to 15/85 for more triggers
- Add fallback RSI entries when CRSI doesn't trigger
- Ensure hold logic maintains position through pullbacks
- Simpler exit: signal reversal or stoploss, not complex conditions

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 7-10 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_simplified_chop_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    CRSI < 15 = oversold (long), CRSI > 85 = overbought (short)
    Relaxed from 10/90 to ensure trades trigger on all symbols
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < max(rsi_period, streak_period, rank_period) + 2:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
            direction[i] = 1
        elif close[i] < close[i-1]:
            if direction[i-1] == -1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = -1
            direction[i] = -1
        else:
            streak[i] = 0
            direction[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_vals > 0)
        down_streaks = np.sum(streak_vals < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50
    
    # Percent Rank of price change
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = 100 * rank
        else:
            percent_rank[i] = 50
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging (mean revert), CHOP < 45 = trending (breakout)
    """
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

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === TREND FILTER (SMA50/200) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        
        # === CONNORS RSI SIGNALS (Relaxed: 15/85) ===
        crsi_oversold = crsi_1d[i] < 15
        crsi_overbought = crsi_1d[i] > 85
        
        # === RSI SIGNALS (fallback) ===
        rsi_oversold = rsi_1d[i] < 30
        rsi_overbought = rsi_1d[i] > 70
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + any bullish filter
            if crsi_oversold:
                if macro_bull or above_sma50 or above_sma200:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE  # Weaker signal without trend
            
            # Short: CRSI overbought + any bearish filter
            if crsi_overbought:
                if macro_bear or below_sma50 or below_sma200:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            
            # Fallback: RSI extremes (ensures trades trigger)
            if desired_signal == 0:
                if rsi_oversold and above_sma200:
                    desired_signal = REDUCED_SIZE
                if rsi_overbought and below_sma200:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish macro + Donchian breakout
            if macro_bull and donchian_breakout_long:
                desired_signal = BASE_SIZE
            # Long pullback: macro bull + CRSI oversold
            elif macro_bull and crsi_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: Bearish macro + Donchian breakout
            if macro_bear and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            # Short pullback: macro bear + CRSI overbought
            elif macro_bear and crsi_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require stronger confluence
            if crsi_oversold and macro_bull and above_sma50:
                desired_signal = REDUCED_SIZE
            if crsi_overbought and macro_bear and below_sma50:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bullish and CRSI not extreme overbought
                if macro_bull and crsi_1d[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bearish and CRSI not extreme oversold
                if macro_bear and crsi_1d[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear + CRSI overbought
            if macro_bear and crsi_1d[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull + CRSI oversold
            if macro_bull and crsi_1d[i] < 25:
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