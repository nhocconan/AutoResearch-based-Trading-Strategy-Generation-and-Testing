#!/usr/bin/env python3
"""
Experiment #614: 4h Primary + 12h HTF — HMA Trend + Connors RSI + Choppiness Regime

Hypothesis: Building on #604 success (4h KAMA+CHOP+RSI, Sharpe=0.378) and current best
mtf_1d_chop_crsi_regime_1w_v1 (Sharpe=0.520), this strategy uses 4h timeframe with 12h
HTF trend filter, combining HMA (faster than KAMA, less lag) with Connors RSI (proven
75% win rate for mean reversion) and Choppiness Index for regime detection.

Key insights from 543 failed strategies:
1. 4h timeframe needs 12h HTF (not 1w - too slow, causes lag)
2. Connors RSI outperforms regular RSI for crypto mean reversion
3. HMA has less lag than KAMA/EMA for trend detection
4. Volume confirmation reduces false breakouts
5. Asymmetric entries: longs need deeper oversold (CRSI<15), shorts moderate (CRSI>85)

Why this might beat Sharpe=0.520:
- 4h captures more opportunities than 1d while avoiding 1h noise
- 12h HMA trend filter keeps us on right side without excessive lag
- Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 has 75% win rate
- Choppiness regime switch: trend-follow when CHOP<45, mean-revert when CHOP>55
- Volume filter (>1.5x 20-bar avg) confirms genuine moves
- Conservative size (0.28) controls drawdown
- 2.5*ATR trailing stop limits losses

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crsi_chop_vol_12h_v1"
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
    CRSI = (RSI(close,3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, streak_abs, 0.0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rsi = 100.0 - (100.0 / (1.0 + streak_avg_gain / (streak_avg_loss + 1e-10)))
    streak_rsi = np.nan_to_num(streak_rsi, nan=50.0)
    
    # Percent Rank of returns
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        if len(window) > 0:
            percent_rank[i] = 100.0 * np.sum(window < returns[i]) / len(window)
    
    # CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Less lag than EMA, smoother than SMA.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume moving average for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(hma_4h_fast[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(vol_ma20[i]) or atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope over 3 bars) ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H HMA CROSSOVER ===
        hma_cross_bull = hma_4h_fast[i] > hma_4h[i]
        hma_cross_bear = hma_4h_fast[i] < hma_4h[i]
        
        # === 4H HMA SLOPE (2 bars) ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-2] if i >= 2 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-2] if i >= 2 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.5 * vol_ma20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0
        is_chop_regime = chop_14[i] > 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 12h trend with 4h HMA crossover entries ---
        if is_trend_regime:
            # LONG: 12h bull + 4h bull cross + price above 12h HMA + CRSI pullback (30-55)
            if hma_12h_slope_bull and hma_cross_bull and price_above_hma_12h:
                if 30.0 <= crsi[i] <= 55.0:
                    if vol_confirmed or hma_4h_slope_bull:
                        new_signal = POSITION_SIZE
            
            # SHORT: 12h bear + 4h bear cross + price below 12h HMA + CRSI bounce (45-70)
            elif hma_12h_slope_bear and hma_cross_bear and price_below_hma_12h:
                if 45.0 <= crsi[i] <= 70.0:
                    if vol_confirmed or hma_4h_slope_bear:
                        new_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at CRSI extremes (asymmetric) ---
        elif is_chop_regime:
            # LONG: CRSI < 15 (extreme oversold) + price below 4h HMA
            if crsi[i] < 15.0 and close[i] < hma_4h[i]:
                new_signal = POSITION_SIZE
            
            # SHORT: CRSI > 85 (extreme overbought) + price above 4h HMA
            elif crsi[i] > 85.0 and close[i] > hma_4h[i]:
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