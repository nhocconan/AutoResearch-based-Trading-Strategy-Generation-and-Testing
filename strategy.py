#!/usr/bin/env python3
"""
Experiment #606: 12h Primary + 1d HTF — KAMA Adaptive Trend + Connors RSI + Choppiness Regime

Hypothesis: Building on #594 success (KAMA+ADX+CHOP on 4h with 12h HTF, Sharpe=0.465) and
#604 refinement (KAMA+CHOP+RSI on 4h, Sharpe=0.378), this strategy moves to 12h primary
with 1d HTF for fewer, higher-quality trades. Key innovations:

1. CONNORS RSI (CRSI) instead of standard RSI — combines RSI(3) + RSI_Streak(2) + PercentRank(100)
   for more sensitive entry timing. Proven 75% win rate in mean-reversion.
2. 1d KAMA for primary trend bias — smoother than HMA, adapts to volatility
3. Choppiness Index regime switch — trend-follow when CHOP<45, mean-revert when CHOP>55
4. Asymmetric position sizing — 0.30 for trend entries, 0.20 for mean-revert (higher risk)
5. 2.5*ATR trailing stoploss — tighter than 3*ATR to protect gains in chop

Why this might beat Sharpe=0.520:
- 12h timeframe = 20-50 trades/year (optimal per Rule 10, less fee drag than 4h)
- Connors RSI catches pullbacks better than standard RSI (3-period vs 14-period)
- 1d HTF trend filter is stronger than 12h (fewer false signals in counter-trend)
- Regime-switching matches market conditions (proven in #594)
- Conservative sizing (0.20-0.30) controls drawdown through 2022 crash

Position sizing: 0.30 trend, 0.20 mean-revert (discrete, max 0.40 per Rule 4)
Target: 20-50 trades/year on 12h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_crsi_chop_1d_v1"
timeframe = "12h"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's price change over last 100 days
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    # Streak = consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (positive streak = gain, negative = loss)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = np.nan_to_num(streak_rsi, nan=50.0)
    
    # Component 3: PercentRank of price change
    pct_change = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = pct_change[i-pr_period:i]
        rank = np.sum(window < pct_change[i]) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    crsi = np.nan_to_num(crsi, nan=50.0)
    
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
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
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
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    # Clip to valid range
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend direction
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h indicators
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_TREND = 0.30
    POSITION_SIZE_MR = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(crsi[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (KAMA slope over 5 bars) ===
        kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-5] if i >= 5 else False
        kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 12H KAMA SLOPE (3 bars) ===
        kama_12h_slope_bull = kama_12h[i] > kama_12h[i-3] if i >= 3 else False
        kama_12h_slope_bear = kama_12h[i] < kama_12h[i-3] if i >= 3 else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0  # Trending
        is_chop_regime = chop_14[i] > 55.0   # Choppy/Range
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 1d trend with CRSI pullback entries ---
        if is_trend_regime:
            # Long: 1d bull slope + 12h bull + price above 1d KAMA + CRSI pullback (30-50)
            if kama_1d_slope_bull and kama_12h_slope_bull and price_above_kama_1d:
                if 25.0 <= crsi[i] <= 50.0:
                    new_signal = POSITION_SIZE_TREND
            
            # Short: 1d bear slope + 12h bear + price below 1d KAMA + CRSI bounce (50-75)
            elif kama_1d_slope_bear and kama_12h_slope_bear and price_below_kama_1d:
                if 50.0 <= crsi[i] <= 75.0:
                    new_signal = -POSITION_SIZE_TREND
        
        # --- CHOP REGIME: Mean reversion at CRSI extremes ---
        elif is_chop_regime:
            # Long: CRSI < 15 (extremely oversold) + price below 1d KAMA
            if crsi[i] < 15.0 and price_below_kama_1d:
                new_signal = POSITION_SIZE_MR
            
            # Short: CRSI > 85 (extremely overbought) + price above 1d KAMA
            elif crsi[i] > 85.0 and price_above_kama_1d:
                new_signal = -POSITION_SIZE_MR
        
        # === HOLD POSITION LOGIC ===
        # If already in position and no new signal, maintain current position
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
        # Exit long if 1d trend flips bear + price below KAMA
        if in_position and position_side > 0:
            if kama_1d_slope_bear and price_below_kama_1d and crsi[i] > 50.0:
                new_signal = 0.0
        
        # Exit short if 1d trend flips bull + price above KAMA
        if in_position and position_side < 0:
            if kama_1d_slope_bull and price_above_kama_1d and crsi[i] < 50.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals