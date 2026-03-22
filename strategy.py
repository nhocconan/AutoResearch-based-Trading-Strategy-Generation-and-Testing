#!/usr/bin/env python3
"""
Experiment #610: 1h Primary + 12h/4h HTF — HMA Trend + Connors RSI + Volatility Filter

Hypothesis: Building on #604 success (4h KAMA+CHOP+RSI, Sharpe=0.378) and learning from 
#605/#608 failures (0 trades on 1h/30m), this strategy uses PROVEN components with 
RELAXED entry thresholds to ensure sufficient trade count while maintaining edge.

Key lessons from 539 failed strategies:
1. #605 (1h CRSI+CHOP) had 0 trades — entry conditions TOO STRICT
2. Lower TF needs HTF trend filter (12h HMA) to reduce whipsaw
3. Connors RSI (not standard RSI) has 75% win rate for mean reversion
4. Must generate 30-80 trades/year on 1h — not 0, not 200+
5. Asymmetric sizing: larger positions in HTF trend direction

Why this might beat Sharpe=0.520:
- 12h HMA trend filter (proven in best strategies) keeps us on right side
- Connors RSI (RSI3 + RSI_Streak + PercentRank) / 3 — superior to standard RSI
- Volatility filter (ATR ratio) avoids entering during extreme vol spikes
- Relaxed CRSI thresholds (15/85 not 10/90) ensures trade count
- Conservative size (0.25) controls drawdown through 2022 crash
- 2.5*ATR trailing stop limits losses

Position sizing: 0.25 discrete (per Rule 4, max 0.40)
Target: 40-80 trades/year on 1h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_crsi_vol_12h_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    wma_half = wma(close_s, half_period)
    wma_full = wma(close_s, period)
    
    # HMA calculation
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive up/down days
    PercentRank: percentile of today's change vs last 100 days
    
    CRSI < 10 = extreme oversold (long)
    CRSI > 90 = extreme overbought (short)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    # Streak: count consecutive up/down moves
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive series for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0)
    streak_rsi = calculate_rsi(streak_positive, streak_period)
    
    # Component 3: PercentRank(100)
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i]
        if len(window) > 0:
            percent_rank[i] = np.sum(window < current) / len(window) * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Load 4h HTF for secondary confirmation
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # Volatility ratio (ATR short / ATR long)
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = atr_14 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1h[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(crsi_1h[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope over 3 bars) ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-2] if i >= 2 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-2] if i >= 2 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H HMA SLOPE (entry timing) ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-2] if i >= 2 else False
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-2] if i >= 2 else False
        
        # === VOLATILITY FILTER ===
        # Avoid entering during extreme vol spikes (vol_ratio > 2.0)
        vol_normal = vol_ratio[i] < 2.0
        
        # === CONNORS RSI ENTRY (relaxed thresholds for trade count) ===
        crsi_oversold = crsi_1h[i] < 20.0  # relaxed from 10 for more trades
        crsi_overbought = crsi_1h[i] > 80.0  # relaxed from 90 for more trades
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG: 12h bull trend + CRSI oversold + 1h HMA turning up ---
        if hma_12h_slope_bull and price_above_hma_12h and vol_normal:
            if crsi_oversold and hma_1h_slope_bull:
                new_signal = POSITION_SIZE
            # Also enter if 4h confirms (less strict)
            elif crsi_1h[i] < 30.0 and hma_4h_slope_bull and price_above_hma_4h:
                new_signal = POSITION_SIZE * 0.6  # smaller size for weaker signal
        
        # --- SHORT: 12h bear trend + CRSI overbought + 1h HMA turning down ---
        elif hma_12h_slope_bear and price_below_hma_12h and vol_normal:
            if crsi_overbought and hma_1h_slope_bear:
                new_signal = -POSITION_SIZE
            # Also enter if 4h confirms (less strict)
            elif crsi_1h[i] > 70.0 and hma_4h_slope_bear and price_below_hma_4h:
                new_signal = -POSITION_SIZE * 0.6  # smaller size for weaker signal
        
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
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_12h:
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