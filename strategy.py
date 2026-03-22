#!/usr/bin/env python3
"""
Experiment #054: 1d Daily Regime-Adaptive with 1w HMA Filter + Connors RSI
Hypothesis: Daily timeframe reduces noise and fee impact. Use 1w HMA for long-term bias.
Key insight: Connors RSI (CRSI) has 75% win rate for mean reversion. Combine with
Choppiness Index to detect regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend.
Asymmetric logic: In bull regime (price>1w HMA), prefer longs on CRSI oversold.
In bear regime (price<1w HMA), prefer shorts on CRSI overbought.
Add Donchian breakout for trend continuation entries when CHOP<38.2.
Position sizing: 0.25 base, 0.35 strong trend, discrete levels. Stoploss 2.5*ATR.
Why this might work: 1d has fewer false signals. CRSI catches reversals. Regime filter
avoids mean-reverting in strong trends. Must generate 10+ trades - loosened CRSI thresholds.
Timeframe: 1d (REQUIRED for exp#054), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_1w_hma_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - very short term
    rsi3 = calculate_rsi(close, rsi_period)
    
    # RSI of Streak - count consecutive up/down days
    streak = np.zeros(n)
    direction = np.zeros(n)  # 1 = up, -1 = down, 0 = neutral
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
            direction[i] = 1
        elif close[i] < close[i-1]:
            if direction[i-1] == -1:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
            direction[i] = -1
        else:
            streak[i] = 0
            direction[i] = 0
    
    # RSI of streak values
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    mask = avg_streak_loss > 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs_streak[mask]))
    rsi_streak[~mask] = 100.0
    
    # Percent Rank - where does current return rank vs last 100 days
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            count_below = np.sum(returns[:-1] < current_return)
            percent_rank[i] = count_below / (len(returns) - 1) * 100
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi3[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(close) if 'close' in dir() else len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bb(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_3 = calculate_rsi(close, 3)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bb(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === LONG-TERM BIAS (1w HMA) ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop[i] > 55  # Loosened from 61.8 for more signals
        trending_regime = chop[i] < 45  # Loosened from 38.2 for more signals
        
        # === CRSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 25  # Loosened from 10 for more trades
        crsi_overbought = crsi[i] > 75  # Loosened from 90 for more trades
        crsi_extreme_oversold = crsi[i] < 15
        crsi_extreme_overbought = crsi[i] > 85
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # === PRICE POSITION ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Bollinger Band position
        near_bb_lower = close[i] <= bb_lower[i] * 1.02 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] >= bb_upper[i] * 0.98 if not np.isnan(bb_upper[i]) else False
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = ema_21[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Donchian breakout
        donchian_breakout_long = close[i] > donchian_upper[i] if not np.isnan(donchian_upper[i]) else False
        donchian_breakout_short = close[i] < donchian_lower[i] if not np.isnan(donchian_lower[i]) else False
        
        # Price pullback to EMA21
        pullback_to_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        pullback_to_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        new_signal = 0.0
        
        # === BULL REGIME (price > 1w HMA) - Prefer Longs ===
        if bull_regime:
            # Mean reversion in ranging market
            if ranging_regime:
                if crsi_oversold or (rsi_oversold and near_bb_lower):
                    new_signal = SIZE_BASE
                elif crsi_extreme_oversold:
                    new_signal = SIZE_STRONG
            
            # Trend following in trending market
            if trending_regime:
                # Pullback entry in uptrend
                if pullback_to_ema21_long and rsi_neutral:
                    if ema_bullish or above_sma200:
                        new_signal = SIZE_BASE
                # Breakout continuation
                if donchian_breakout_long and rsi[i] < 70:
                    new_signal = SIZE_HALF
        
        # === BEAR REGIME (price < 1w HMA) - Prefer Shorts ===
        if bear_regime:
            # Mean reversion in ranging market
            if ranging_regime:
                if crsi_overbought or (rsi_overbought and near_bb_upper):
                    new_signal = -SIZE_BASE
                elif crsi_extreme_overbought:
                    new_signal = -SIZE_STRONG
            
            # Trend following in trending market
            if trending_regime:
                # Pullback entry in downtrend
                if pullback_to_ema21_short and rsi_neutral:
                    if ema_bearish or below_sma200:
                        new_signal = -SIZE_BASE
                # Breakdown continuation
                if donchian_breakout_short and rsi[i] > 30:
                    new_signal = -SIZE_HALF
        
        # === TRANSITION REGIME (neutral chop) - Use CRSI extremes only ===
        if not ranging_regime and not trending_regime:
            if crsi_extreme_oversold and bull_regime:
                new_signal = SIZE_HALF
            if crsi_extreme_overbought and bear_regime:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals