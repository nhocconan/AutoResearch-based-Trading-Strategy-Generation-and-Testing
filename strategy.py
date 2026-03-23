#!/usr/bin/env python3
"""
Experiment #067: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: Daily timeframe with weekly HTF trend filter using Connors RSI for mean reversion
entries will work better in bear/range markets (2025 test period). Connors RSI has proven
75% win rate for counter-trend entries. Choppiness Index filters regime to avoid mean
reversion in strong trends. Weekly HMA provides macro bias to avoid counter-trend trades.

Key innovations:
1) Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2) Weekly HMA trend filter: only long when price > 1w HMA, only short when price < 1w HMA
3) Choppiness Index: CHOP > 50 = range (enable mean reversion), CHOP < 40 = trend (disable MR)
4) Relaxed CRSI thresholds: <25 for long, >75 for short (ensure sufficient trades)
5) ATR-based stoploss: 2.5*ATR trailing stop
6) Position size: 0.30 for all entries (discrete, minimizes churn)

Why this should work on 1d:
- Daily bars = fewer false signals, lower fee drag
- Connors RSI excels at catching pullbacks in bull markets and bounces in bear markets
- Weekly HTF prevents counter-trend trades during strong moves
- Choppiness filter avoids mean reversion during trending periods (where it fails)
- 2025 bear market favors mean reversion over trend following

Position size: 0.30 (discrete)
Stoploss: 2.5*ATR trailing
Target: 25-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_regime_1w_v1"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
    delta = close_s.diff().values
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, 50.0)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_window > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    
    # Percent Rank component
    pr = np.full(n, 50.0)
    for i in range(pr_period, n):
        returns = np.diff(close[i-pr_period:i+1]) / close[i-pr_period:i]
        if len(returns) > 0:
            current_return = (close[i] - close[i-1]) / close[i-1]
            pr[i] = (np.sum(returns < current_return) / len(returns)) * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + pr) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_3 = calculate_rsi(close, period=3)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if atr_14[i] == 0 or np.isnan(atr_14[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Range market - enable mean reversion
        is_trending = chop_value < 40.0  # Trend market - disable mean reversion
        
        # === CONNORS RSI SIGNALS (Relaxed thresholds for more trades) ===
        crsi_oversold = crsi[i] < 25.0  # Long entry
        crsi_overbought = crsi[i] > 75.0  # Short entry
        
        # === RSI CONFIRMATION ===
        rsi_extreme_low = rsi_3[i] < 20.0
        rsi_extreme_high = rsi_3[i] > 80.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Mean Reversion Long: CRSI oversold + ranging market + not strongly bearish HTF
        if crsi_oversold and is_ranging:
            if price_above_hma_1w or (not price_below_hma_1w):
                new_signal = POSITION_SIZE
        
        # Mean Reversion Short: CRSI overbought + ranging market + not strongly bullish HTF
        elif crsi_overbought and is_ranging:
            if price_below_hma_1w or (not price_above_hma_1w):
                new_signal = -POSITION_SIZE
        
        # Trending market: only enter with HTF trend (more conservative)
        if is_trending:
            if crsi_oversold and price_above_hma_1w and rsi_extreme_low:
                new_signal = POSITION_SIZE
            elif crsi_overbought and price_below_hma_1w and rsi_extreme_high:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC (avoid churn) ===
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0:
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
        
        # === EXIT ON HTF TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1w and chop_value < 45.0:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w and chop_value < 45.0:
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