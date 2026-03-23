#!/usr/bin/env python3
"""
Experiment #641: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 567 failed strategies, the pattern is clear:
1. Too many filters = 0 trades (#632, #635, #638 all had Sharpe=0.000)
2. 4h timeframe CAN work but needs SIMPLER entry conditions
3. Connors RSI has 75% win rate in research but thresholds were too extreme
4. Choppiness Index regime detection worked well in #637 (Sharpe=0.257 on 1d)

This strategy uses:
- 4h primary timeframe (target 20-50 trades/year per Rule 10)
- 1d HTF for major trend direction (HMA slope)
- Connors RSI (CRSI) with RELAXED thresholds (≤20 long, ≥80 short) to ensure trades
- Choppiness Index regime: CHOP>61.8 = mean revert, CHOP<38.2 = trend follow
- Simple SMA200 filter for long/short bias
- 2.5*ATR trailing stoploss
- Discrete position size 0.30

Why this might beat Sharpe=0.520:
- Fewer conflicting filters = more trades (critical after 0-trade failures)
- CRSI thresholds relaxed from ≤10/≥90 to ≤20/≥80 (still selective but achievable)
- Choppiness regime switch adapts to market conditions
- 1d HTF trend filter keeps us on right side of major moves
- 4h entry timing captures pullbacks within daily trend

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 30-50 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d_v2"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on up/down streak length
    PercentRank(100): Percentile rank of today's return over last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI: count consecutive up/down days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak values (convert to positive for RSI calc)
    streak_positive = streak + np.abs(streak.min()) + 1
    rsi_streak = calculate_rsi(streak_positive, streak_period)
    
    # PercentRank(100): percentile of today's return vs last 100
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period:i]
        today_return = returns.iloc[i]
        rank = np.sum(window_returns < today_return) / rank_period
        percent_rank[i] = rank * 100.0
    
    # CRSI = average of 3 components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    chop = np.zeros(len(close))
    mask = price_range > 0
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(sma_200[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 3 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop[i] > 61.8  # Mean reversion mode
        trending_regime = chop[i] < 38.2  # Trend following mode
        
        # === CONNORS RSI SIGNALS (RELAXED thresholds for trade generation) ===
        crsi_oversold = crsi[i] <= 20.0  # Long signal (relaxed from ≤10)
        crsi_overbought = crsi[i] >= 80.0  # Short signal (relaxed from ≥90)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Choppy market (mean reversion)
        # Entry: CRSI oversold + price above SMA200 (bullish bias)
        if choppy_regime:
            if crsi_oversold and price_above_sma200:
                new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (trend following)
        # Entry: CRSI oversold + 1d HMA bullish + price above 1d HMA
        elif trending_regime:
            if crsi_oversold and hma_1d_slope_bull and price_above_hma_1d:
                new_signal = POSITION_SIZE
        
        # Fallback: Any regime with strong CRSI signal + SMA200 filter
        # This ensures we get trades even when regime detection is ambiguous
        if new_signal == 0.0:
            if crsi_oversold and price_above_sma200 and price_above_hma_1d:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Choppy market (mean reversion)
        # Entry: CRSI overbought + price below SMA200 (bearish bias)
        if choppy_regime:
            if crsi_overbought and price_below_sma200:
                new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (trend following)
        # Entry: CRSI overbought + 1d HMA bearish + price below 1d HMA
        elif trending_regime:
            if crsi_overbought and hma_1d_slope_bear and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # Fallback: Any regime with strong CRSI signal + SMA200 filter
        if new_signal == 0.0:
            if crsi_overbought and price_below_sma200 and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals