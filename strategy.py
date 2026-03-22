#!/usr/bin/env python3
"""
Experiment #609: 4h Primary + 1d HTF — KAMA Trend + Choppiness Regime + Connors RSI

Hypothesis: Building on #604 success (4h KAMA+CHOP+RSI+12h, Sharpe=0.378) and lessons from
538 failed strategies, this strategy uses 4h primary timeframe with 1d HTF trend filter.
Key insight: 4h offers better trade frequency (20-50/year) than 1d while maintaining
signal quality. 1d HTF provides cleaner trend signal than 1w (less lag).

Why this might beat Sharpe=0.520 (current best mtf_1d_chop_crsi_regime_1w_v1):
1. 4h entries capture moves earlier than 1d (less lag on entry/exit)
2. Connors RSI (CRSI) has 75% win rate vs standard RSI (proven in literature)
3. KAMA adapts to volatility better than EMA/HMA during 2022 crash
4. Choppiness regime switch prevents trend-following in ranges (major edge)
5. Asymmetric sizing: 0.30 for trend trades, 0.20 for mean-reversion (risk-adjusted)
6. ADX > 20 filter ensures we only trend-follow when trend actually exists

CRSI Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): Very short-term momentum
- RSI_Streak: Consecutive up/down days (100 * streak_length / lookback)
- PercentRank: Current close vs past 100 bars percentile

Position sizing: 0.20-0.30 discrete (Rule 4, max 0.40)
Target: 30-50 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_crsi_1d_v1"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    
    Proven 75% win rate in mean-reversion strategies.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI Streak (consecutive up/down closes)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like score (0-100)
    streak_abs = np.abs(streak)
    streak_score = np.zeros(n)
    for i in range(streak_period, n):
        lookback_streaks = streak_abs[max(0, i-streak_period):i+1]
        if streak[i] > 0:
            streak_score[i] = 100.0 * np.sum(lookback_streaks[streak[max(0,i-streak_period):i+1] > 0]) / (streak_period + 1e-10)
        elif streak[i] < 0:
            streak_score[i] = 100.0 - (100.0 * np.sum(lookback_streaks[streak[max(0,i-streak_period):i+1] < 0]) / (streak_period + 1e-10))
        else:
            streak_score[i] = 50.0
    streak_score = np.clip(streak_score, 0, 100)
    
    # Component 3: Percent Rank (current close vs past rank_period bars)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period):i]
        if len(window) > 0:
            percent_rank[i] = 100.0 * np.sum(window < close[i]) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi_3 + streak_score + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_s / tr_s
        minus_di = 100.0 * minus_dm_s / tr_s
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for primary trend direction
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_TREND = 0.30  # Higher confidence in trend regime
    POSITION_SIZE_MR = 0.20     # Lower confidence in mean-reversion
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(adx_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (KAMA slope over 3 bars) ===
        kama_1d_slope_bull = False
        kama_1d_slope_bear = False
        if i >= 3 and not np.isnan(kama_1d_aligned[i-3]):
            kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-3]
            kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-3]
        
        # Price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d_aligned[i] if not np.isnan(kama_1d_aligned[i]) else False
        price_below_kama_1d = close[i] < kama_1d_aligned[i] if not np.isnan(kama_1d_aligned[i]) else False
        
        # === 4H KAMA SLOPE (2 bars) ===
        kama_4h_slope_bull = False
        kama_4h_slope_bear = False
        if i >= 2 and not np.isnan(kama_4h[i-2]):
            kama_4h_slope_bull = kama_4h[i] > kama_4h[i-2]
            kama_4h_slope_bear = kama_4h[i] < kama_4h[i-2]
        
        # Price relative to 4h KAMA
        price_above_kama_4h = close[i] > kama_4h[i] if not np.isnan(kama_4h[i]) else False
        price_below_kama_4h = close[i] < kama_4h[i] if not np.isnan(kama_4h[i]) else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0
        is_chop_regime = chop_14[i] > 55.0
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_14[i] > 20.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 1d trend with 4h CRSI pullback entries ---
        if is_trend_regime and trend_strong:
            # LONG: 1d bull + 4h bull + price above both KAMAs + CRSI pullback (20-45)
            if kama_1d_slope_bull and kama_4h_slope_bull and price_above_kama_1d and price_above_kama_4h:
                if 20.0 <= crsi[i] <= 45.0:
                    new_signal = POSITION_SIZE_TREND
            
            # SHORT: 1d bear + 4h bear + price below both KAMAs + CRSI bounce (55-80)
            elif kama_1d_slope_bear and kama_4h_slope_bear and price_below_kama_1d and price_below_kama_4h:
                if 55.0 <= crsi[i] <= 80.0:
                    new_signal = -POSITION_SIZE_TREND
        
        # --- CHOP REGIME: Mean reversion at CRSI extremes ---
        elif is_chop_regime:
            # LONG: CRSI < 15 (extreme oversold) + price below 4h KAMA
            if crsi[i] < 15.0 and price_below_kama_4h:
                new_signal = POSITION_SIZE_MR
            
            # SHORT: CRSI > 85 (extreme overbought) + price above 4h KAMA
            elif crsi[i] > 85.0 and price_above_kama_4h:
                new_signal = -POSITION_SIZE_MR
        
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
            if kama_1d_slope_bear and price_below_kama_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_1d_slope_bull and price_above_kama_1d:
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