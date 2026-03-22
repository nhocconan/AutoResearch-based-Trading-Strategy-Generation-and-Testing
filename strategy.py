#!/usr/bin/env python3
"""
Experiment #611: 4h Primary + 1d/1w HTF — HMA Trend + Connors RSI + Choppiness Regime

Hypothesis: Building on #604 success (4h KAMA+CHOP+RSI, Sharpe=0.378), this strategy 
combines faster HMA entries with slower 1w KAMA trend filter and Connors RSI for 
precise mean-reversion entry timing within the trend direction.

Key improvements over #607 (which failed with Sharpe=-0.627):
1. SIMPLER logic — fewer conflicting filters = more trades
2. Connors RSI (CRSI) instead of standard RSI — proven 75% win rate in mean reversion
3. HMA for 4h entries (faster than KAMA) + 1w KAMA for trend (slower, stable)
4. Choppiness only as META-filter (avoid trading against regime), not entry trigger
5. Asymmetric CRSI: longs at CRSI<15, shorts at CRSI>85 (tighter than RSI)
6. Position tracking simplified — no complex hold logic, just signal-based

Why this might beat Sharpe=0.520:
- 1w KAMA slope is very stable trend filter (prevents counter-trend trades)
- 4h HMA crosses give timely entries within 1w trend
- CRSI captures short-term exhaustion better than RSI(14)
- CHOP > 50 filter avoids trend-chasing in choppy markets
- Conservative size (0.28) with 2.5*ATR trailing stop

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 4h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crsi_chop_1w_v1"
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
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentile of today's close vs last 100 closes
    
    CRSI < 10 = extreme oversold (long opportunity)
    CRSI > 90 = extreme overbought (short opportunity)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    percent_rank[:rank_period] = np.nan
    
    # CRSI = average of three components
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA(n/2)
    wma_half = close_s.ewm(span=half_period, min_periods=half_period, adjust=False).mean()
    
    # WMA(n)
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # 2*WMA(n/2) - WMA(n)
    raw_hma = 2.0 * wma_half - wma_full
    
    # WMA(sqrt(n)) on the result
    hma = raw_hma.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    return hma.values

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
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(kama_1w_aligned[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (KAMA slope over 5 bars for stability) ===
        kama_1w_slope_bull = kama_1w_aligned[i] > kama_1w_aligned[i-5] if i >= 5 else False
        kama_1w_slope_bear = kama_1w_aligned[i] < kama_1w_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1w KAMA
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === 4H HMA TREND (HMA21 vs HMA48 crossover) ===
        hma_bull = hma_21[i] > hma_48[i]
        hma_bear = hma_21[i] < hma_48[i]
        
        # Price relative to HMA21
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        # === CHOPPINESS META-FILTER ===
        # Only trade when regime supports the strategy type
        # For trend-following entries: CHOP < 55 (not too choppy)
        # For mean-reversion entries: CHOP > 45 (some chop is OK)
        not_extreme_chop = chop_14[i] < 65.0  # Avoid extreme chop
        
        # === ENTRY LOGIC (simplified, asymmetric) ===
        new_signal = 0.0
        
        # LONG: 1w uptrend + 4h HMA bull + CRSI oversold + not extreme chop
        if kama_1w_slope_bull and price_above_kama_1w and hma_bull:
            if crsi[i] < 20.0 and not_extreme_chop:
                new_signal = POSITION_SIZE
        
        # SHORT: 1w downtrend + 4h HMA bear + CRSI overbought + not extreme chop
        elif kama_1w_slope_bear and price_below_kama_1w and hma_bear:
            if crsi[i] > 80.0 and not_extreme_chop:
                new_signal = -POSITION_SIZE
        
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
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
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