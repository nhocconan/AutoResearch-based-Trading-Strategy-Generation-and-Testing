#!/usr/bin/env python3
"""
Experiment #952: 12h Primary + 1d/1w HTF — Dual Regime (Mean Revert + Trend Follow)

Hypothesis: After 681+ failed strategies, the key is SIMPLER entry conditions that
actually generate trades on ALL symbols. Complex multi-filter strategies (funding +
choppiness + vol + BB + RSI) result in 0 trades.

Key insights from research:
1. Connors RSI (CRSI) for mean reversion: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long when CRSI < 15, Short when CRSI > 85. 75% win rate in range markets.
2. Choppiness Index regime: CHOP > 55 = range (use CRSI), CHOP < 45 = trend (use breakout)
3. Donchian breakout for trend: 55-period high/low break + HMA(21) confirmation
4. 1d HMA(21) for macro bias — only trade in direction of daily trend
5. 1w HMA(21) for secular regime filter

Why 12h timeframe:
- Target 20-50 trades/year (minimal fee drag)
- HTF signals (1d/1w) provide strong trend bias
- Proven patterns: ETH Sharpe +0.923, SOL Sharpe +0.782 with similar logic
- Less noise than 4h/1h, clearer regime detection

Critical improvements over failed strategies:
- RELAXED entry thresholds (CRSI < 20 not < 10, CHOP > 50 not > 61.8)
- Funding rate REMOVED as primary filter (caused 0 trades when data missing)
- Dual regime: mean revert in chop, trend follow otherwise
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- MUST generate trades on BTC/ETH/SOL individually

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d1w_hma_atr_v1"
timeframe = "12h"
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
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    delta = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i-1] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        if not np.isnan(streak[i]):
            # Simple mapping: positive streak = high, negative = low
            streak_rsi[i] = 50 + np.clip(streak[i] * 10, -50, 50)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = (rank / (rank_period - 1)) * 100
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
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

def calculate_donchian(high, low, period=55):
    """Donchian Channel — highest high and lowest low over period."""
    n = len(close)
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels for breakout detection
    donch_upper_55, donch_lower_55 = calculate_donchian(high, low, period=55)
    donch_upper_20, donch_lower_20 = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donch_upper_55[i]) or np.isnan(donch_lower_55[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 50
        trending_regime = chop_12h[i] < 45
        
        # === CONNORS RSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_12h[i] < 20
        crsi_overbought = crsi_12h[i] > 80
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        
        # === DONCHIAN BREAKOUT SIGNALS (Trend Following) ===
        donch_breakout_long = close[i] > donch_upper_55[i-1] if not np.isnan(donch_upper_55[i-1]) else False
        donch_breakout_short = close[i] < donch_lower_55[i-1] if not np.isnan(donch_lower_55[i-1]) else False
        donch_breakout_long_20 = close[i] > donch_upper_20[i-1] if not np.isnan(donch_upper_20[i-1]) else False
        donch_breakout_short_20 = close[i] < donch_lower_20[i-1] if not np.isnan(donch_lower_20[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI oversold + macro/medium trend support
            if crsi_oversold and (macro_bull or trend_1d_bullish):
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold (stronger signal, trade regardless of trend)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + macro/medium trend support
            if crsi_overbought and (macro_bear or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought (stronger signal, trade regardless of trend)
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Donchian ===
        elif trending_regime:
            # Long: Donchian breakout + bullish trend alignment
            if donch_breakout_long and (macro_bull or trend_1d_bullish):
                desired_signal = BASE_SIZE
            # Long: Shorter Donchian breakout + strong trend
            elif donch_breakout_long_20 and macro_bull and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            # Short: Donchian breakout + bearish trend alignment
            if donch_breakout_short and (macro_bear or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            # Short: Shorter Donchian breakout + strong trend
            elif donch_breakout_short_20 and macro_bear and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 50) ===
        else:
            # Conservative: CRSI extremes only
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Or trend with strong confirmation
            if donch_breakout_long and macro_bull and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            if donch_breakout_short and macro_bear and trend_1d_bearish:
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
                # Hold long if trend intact and CRSI not overbought
                if (macro_bull or trend_1d_bullish) and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (macro_bear or trend_1d_bearish) and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + medium trend reverses + CRSI overbought
            if macro_bear and trend_1d_bearish and crsi_12h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + medium trend reverses + CRSI oversold
            if macro_bull and trend_1d_bullish and crsi_12h[i] < 25:
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