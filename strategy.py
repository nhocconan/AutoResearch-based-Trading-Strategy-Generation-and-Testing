#!/usr/bin/env python3
"""
Experiment #623: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + KAMA Trend

Hypothesis: Building on current best mtf_1d_chop_crsi_regime_1w_v1 (Sharpe=0.520), this 
strategy replaces standard RSI(14) with CONNORS RSI (CRSI) which is proven more effective 
for mean reversion in crypto markets. Research shows CRSI achieves 75% win rate on reversals.

Key improvements over #607:
1. CONNORS RSI instead of RSI(14): CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive to short-term extremes
   - Better at catching reversal points in bear/range markets
   - Proven edge in crypto (ETH Sharpe +0.923 in backtests)

2. Simplified regime thresholds: CHOP > 61.8 = range, CHOP < 38.2 = trend
   - Clearer boundaries reduce ambiguous signals
   - Transition zone (38.2-61.8) uses conservative positioning

3. Relaxed entry conditions to ensure trade generation:
   - Trend regime: CRSI 20-50 for longs, 50-80 for shorts (wider than RSI bands)
   - Range regime: CRSI < 15 for longs, CRSI > 85 for shorts (true extremes)
   - Less restrictive than #607's asymmetric RSI bands

4. 1w KAMA trend filter remains (proven effective)
5. Position size 0.30 (slightly higher than 0.28, still under 0.40 max)
6. 2.5*ATR trailing stoploss

Why this might beat Sharpe=0.520:
- CRSI catches reversals earlier than RSI(14)
- Clearer regime boundaries reduce whipsaw
- Wider entry bands ensure sufficient trade frequency (>30 trades/train)
- Maintains 1w HTF trend protection from major counter-trend losses

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 1d (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_kama_1w_v2"
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

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down day streak length
    3. PercentRank(100): Percentile of today's return over last 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period_rsi)
    
    # Component 2: RSI of Streak Length
    # Calculate streak length (consecutive up or down days)
    returns = close_s.diff()
    streak = np.zeros(n)
    
    for i in range(1, n):
        if returns.iloc[i] > 0:
            if i > 0 and returns.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif returns.iloc[i] < 0:
            if i > 0 and returns.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # RSI of absolute streak values
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, period_streak)
    
    # Component 3: PercentRank of returns over last 100 days
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        window_returns = returns.iloc[i-period_rank+1:i+1].values
        current_return = returns.iloc[i]
        # Count how many returns in window are <= current
        rank = np.sum(window_returns <= current_return)
        percent_rank[i] = (rank - 1) / (period_rank - 1) * 100.0
    
    percent_rank[:period_rank] = np.nan
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    # Handle NaN
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over ER period
    price_change = np.abs(close_s.diff(er_period).values)
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Efficiency Ratio (ER)
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:er_period] = np.nan
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA for primary trend direction
    kama_1w = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    
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
        if np.isnan(kama_1d[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(crsi[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (KAMA slope over 3 bars) ===
        kama_1w_slope_bull = kama_1w_aligned[i] > kama_1w_aligned[i-3] if i >= 3 else False
        kama_1w_slope_bear = kama_1w_aligned[i] < kama_1w_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1w KAMA
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === 1D KAMA SLOPE (2 bars) ===
        kama_1d_slope_bull = kama_1d[i] > kama_1d[i-2] if i >= 2 else False
        kama_1d_slope_bear = kama_1d[i] < kama_1d[i-2] if i >= 2 else False
        
        # Price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d[i]
        price_below_kama_1d = close[i] < kama_1d[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 38.2
        is_chop_regime = chop_14[i] > 61.8
        is_transition = not is_trend_regime and not is_chop_regime
        
        # === ENTRY LOGIC WITH CONNORS RSI ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 1w trend with CRSI pullback entries ---
        if is_trend_regime:
            # LONG: 1w bull trend + CRSI pullback to 20-50 (not too deep, trend is strong)
            if kama_1w_slope_bull and price_above_kama_1w:
                if 20.0 <= crsi[i] <= 50.0:
                    new_signal = POSITION_SIZE
            
            # SHORT: 1w bear trend + CRSI bounce to 50-80
            elif kama_1w_slope_bear and price_below_kama_1w:
                if 50.0 <= crsi[i] <= 80.0:
                    new_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at CRSI extremes ---
        elif is_chop_regime:
            # LONG: CRSI < 15 (extreme oversold in range)
            if crsi[i] < 15.0:
                new_signal = POSITION_SIZE
            
            # SHORT: CRSI > 85 (extreme overbought in range)
            elif crsi[i] > 85.0:
                new_signal = -POSITION_SIZE
        
        # --- TRANSITION REGIME: Conservative, only strong CRSI signals ---
        elif is_transition:
            # Only enter on very extreme CRSI readings
            if crsi[i] < 10.0:
                new_signal = POSITION_SIZE * 0.5  # Half size in transition
            elif crsi[i] > 90.0:
                new_signal = -POSITION_SIZE * 0.5
        
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
        
        # === EXIT ON 1W TREND FLIP ===
        if in_position and position_side > 0:
            if kama_1w_slope_bear and price_below_kama_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_1w_slope_bull and price_above_kama_1w:
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