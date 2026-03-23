#!/usr/bin/env python3
"""
Experiment #769: 4h Primary + 1d HTF — Choppiness Index Regime + Connors RSI + HMA Trend

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. Choppiness Index (CHOP) is proven regime filter for crypto (61.8/38.2 thresholds)
2. 1d EMA50 is more stable trend filter than 12h for 4h entries
3. Volume filter too restrictive — removing it increases trade frequency
4. CRSI thresholds <20/>80 generate more trades than <15/>85 while maintaining edge
5. Simpler hold logic reduces overfitting — exit on opposite signal or stoploss only
6. HMA(21) on 1d provides faster trend response than EMA50

Strategy design:
1. 1d HMA(21) for trend bias (aligned via mtf_data helper)
2. 4h Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. 4h Choppiness Index(14) for regime (CHOP>61.8=ranging, CHOP<38.2=trending)
4. 4h ATR(14) for trailing stop (2.5x)
5. 4h Bollinger Bands(20,2.0) for mean reversion bounds
6. Discrete signals: 0.0, ±0.25, ±0.30
7. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key improvements from #764:
- Replaced 12h EMA50 with 1d HMA21 (more stable, faster response)
- Replaced ADX hysteresis with Choppiness Index (proven crypto edge)
- Removed volume filter (was causing 0 trades on some symbols)
- Relaxed CRSI thresholds (<20/>80 vs <15/>85) for more trade frequency
- Simplified hold/exit logic (reduces overfitting)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_hma_1d_bb_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """
    Hull Moving Average — faster response, less lag than EMA.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(series).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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
    """
    Connors RSI Streak component.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <20 = oversold, >80 = overbought.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = ranging/choppy (mean reversion)
    CHOP < 38.2 = trending (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    atr_vals = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.nansum(atr_vals[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(chop_4h[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_4h[i] > 61.8
        trending_regime = chop_4h[i] < 38.2
        neutral_regime = not ranging_regime and not trending_regime
        
        # === CRSI SIGNALS (relaxed thresholds for more trades) ===
        crsi_oversold = crsi_4h[i] < 20
        crsi_overbought = crsi_4h[i] > 80
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        crsi_neutral_low = 25 < crsi_4h[i] < 45
        crsi_neutral_high = 55 < crsi_4h[i] < 75
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        near_bb_lower = close[i] < bb_sma[i] * 0.99
        near_bb_upper = close[i] > bb_sma[i] * 1.01
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) — Mean Reversion ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + below BB lower
            if crsi_oversold and below_bb_lower:
                desired_signal = BASE_SIZE if trend_1d_bullish else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + above BB upper
            if crsi_overbought and above_bb_upper:
                desired_signal = -BASE_SIZE if trend_1d_bearish else -REDUCED_SIZE
            
            # Conservative: extreme CRSI only
            if crsi_extreme_oversold and not trend_1d_bearish:
                if desired_signal <= 0:
                    desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and not trend_1d_bullish:
                if desired_signal >= 0:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) — Trend Following ===
        elif trending_regime:
            # Trend pullback long: 1d bullish + CRSI neutral low + near BB lower
            if trend_1d_bullish and crsi_neutral_low:
                if near_bb_lower or below_bb_lower:
                    desired_signal = BASE_SIZE
            
            # Trend pullback short: 1d bearish + CRSI neutral high + near BB upper
            if trend_1d_bearish and crsi_neutral_high:
                if near_bb_upper or above_bb_upper:
                    desired_signal = -BASE_SIZE
            
            # Breakout continuation
            if trend_1d_bullish and above_bb_upper and crsi_4h[i] < 70:
                desired_signal = BASE_SIZE
            
            if trend_1d_bearish and below_bb_lower and crsi_4h[i] > 30:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only extreme CRSI + trend alignment
            if crsi_extreme_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Moderate CRSI with strong BB position
            if crsi_oversold and below_bb_lower and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_overbought and above_bb_upper and trend_1d_bearish:
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
                if trend_1d_bullish and crsi_4h[i] < 85:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if trend_1d_bearish and crsi_4h[i] > 15:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses strongly or CRSI very overbought
            if trend_1d_bearish and crsi_4h[i] > 75:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper and crsi_4h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses strongly or CRSI very oversold
            if trend_1d_bullish and crsi_4h[i] < 25:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower and crsi_4h[i] < 30:
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