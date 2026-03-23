#!/usr/bin/env python3
"""
Experiment #413: 1d Primary + 1w HTF — KAMA Trend + Choppiness Regime + Connors RSI

Hypothesis: Daily timeframe with weekly bias filter provides optimal balance between
trade frequency (20-50/year) and signal quality. KAMA adapts to volatility better than
HMA/EMA, reducing whipsaw in choppy conditions. Connors RSI (3-period) catches short-term
extremes more effectively than standard RSI(14). Choppiness Index regime detection
switches between trend-following (CHOP<38.2) and mean-reversion (CHOP>61.8) modes.

Key components:
1. KAMA(10,2,30) - Kaufman Adaptive MA for trend direction
2. Choppiness Index(14) - Regime detection (trend vs range)
3. Connors RSI(3,2,100) - Entry timing with 3-component calculation
4. 1w HMA(21) - Weekly bias filter (only trade with HTF trend)
5. ATR(14) trailing stop (2.5x) - Risk management
6. Position sizing: 0.0, ±0.25, ±0.30 (discrete levels)

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_crsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate change and noise
    change = np.abs(close - np.roll(close, efficiency_period))
    change[:efficiency_period] = np.nan
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i-1] + np.abs(close[i] - close[i-1])
    noise[:efficiency_period] = np.nan
    
    # Efficiency Ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (noise + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[efficiency_period] = close[efficiency_period]
    
    # Calculate KAMA
    for i in range(efficiency_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): 3-period RSI on price
    RSI(streak, 2): 2-period RSI on up/down streak length
    PercentRank(100): Percentile rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3) on close
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.rolling(window=streak_period, min_periods=streak_period).mean()
    streak_avg_loss = streak_loss.rolling(window=streak_period, min_periods=streak_period).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank(100)
    returns = close_s.pct_change() * 100
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if len(window) > 0:
            percent_rank.iloc[i] = (window < current).sum() / len(window) * 100
        else:
            percent_rank.iloc[i] = 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_close.iloc[i]) and not np.isnan(rsi_streak.iloc[i]) and not np.isnan(percent_rank.iloc[i]):
            crsi[i] = (rsi_close.iloc[i] + rsi_streak.iloc[i] + percent_rank.iloc[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    kama = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 1d
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (KAMA slope) ===
        kama_bullish = kama[i] > kama[i-5] if i >= 5 else False
        kama_bearish = kama[i] < kama[i-5] if i >= 5 else False
        
        # === PRICE vs KAMA ===
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === CONNORS RSI THRESHOLDS ===
        # CRSI < 10 = extremely oversold (long opportunity)
        # CRSI > 90 = extremely overbought (short opportunity)
        # CRSI < 30 = oversold (moderate long)
        # CRSI > 70 = overbought (moderate short)
        crsi_extreme_long = crsi[i] < 15.0
        crsi_extreme_short = crsi[i] > 85.0
        crsi_moderate_long = crsi[i] < 35.0
        crsi_moderate_short = crsi[i] > 65.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        if price_above_hma_1w:  # HTF bullish bias required for longs
            if is_trending and kama_bullish and price_above_kama:
                # Trend following mode - enter on pullback
                if crsi_moderate_long:
                    desired_signal = position_size
                elif crsi_extreme_long:
                    desired_signal = position_size * 1.2  # Larger size on extreme
            elif is_choppy:
                # Mean reversion in range - buy oversold
                if crsi_extreme_long or crsi_moderate_long:
                    desired_signal = position_size
            elif kama_bullish and price_above_kama:
                # KAMA bullish pullback
                if crsi_moderate_long:
                    desired_signal = position_size
        
        # SHORT SETUP
        if price_below_hma_1w:  # HTF bearish bias required for shorts
            if is_trending and kama_bearish and price_below_kama:
                # Trend following mode - enter on rally
                if crsi_moderate_short:
                    desired_signal = -position_size
                elif crsi_extreme_short:
                    desired_signal = -position_size * 1.2
            elif is_choppy:
                # Mean reversion in range - sell overbought
                if crsi_extreme_short or crsi_moderate_short:
                    desired_signal = -position_size
            elif kama_bearish and price_below_kama:
                # KAMA bearish rally
                if crsi_moderate_short:
                    desired_signal = -position_size
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === CRSI EXTREME EXIT ===
        if in_position and position_side > 0 and crsi[i] > 90.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 10.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1w:
                desired_signal = position_size
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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