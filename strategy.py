#!/usr/bin/env python3
"""
Experiment #411: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Bias

Hypothesis: Previous strategies used standard RSI which is too slow for mean reversion.
Connors RSI (CRSI) is proven for short-term mean reversion with 75% win rate in literature.
Combined with Choppiness Index regime filter and 1d HMA bias, this should:
1. Generate MORE trades than overly strict Donchian breakout strategies (#399, #402 failed)
2. Work in BOTH trending and ranging markets (regime-adaptive)
3. Use 1d HTF for directional bias (proven in #409 Sharpe=0.311)
4. Exit quickly on mean reversion (CRSI extremes reverse fast)

Key innovations vs #409:
- Connors RSI instead of standard RSI (faster mean reversion signal)
- Streak-based RSI component captures momentum exhaustion
- PercentRank component normalizes across different vol regimes
- Simpler regime logic (CHOP > 50 = range, < 50 = trend)
- More aggressive position sizing on high-confidence CRSI extremes

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_crsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI(Streak): RSI of consecutive up/down bars (momentum exhaustion)
    PercentRank: Percentile of today's return over last N periods
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period_rsi)
    
    # Component 2: Streak RSI
    # Streak = consecutive up (+) or down (-) bars
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (use absolute values for RSI calculation)
    streak_positive = np.maximum(streak, 0)
    streak_negative = np.maximum(-streak, 0)
    
    avg_streak_gain = pd.Series(streak_positive).ewm(span=period_streak, min_periods=period_streak, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_negative).ewm(span=period_streak, min_periods=period_streak, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank of returns
    returns = close_s.pct_change().values
    percent_rank = np.full(n, np.nan)
    
    for i in range(period_rank, n):
        window = returns[i-period_rank+1:i+1]
        if np.all(np.isnan(window)):
            percent_rank[i] = 50.0
        else:
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                current_return = returns[i]
                if np.isnan(current_return):
                    percent_rank[i] = 50.0
                else:
                    rank = np.sum(valid_window <= current_return)
                    percent_rank[i] = 100.0 * rank / len(valid_window)
            else:
                percent_rank[i] = 50.0
    
    # Combine components
    for i in range(period_rank, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    crsi = calculate_crsi(close, period_rsi=3, period_streak=2, period_rank=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[150:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h
    MAX_SIZE = 0.35   # Max position on extreme CRSI
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 50.0  # Simplified: >50 = range, <50 = trend
        is_trending = chop[i] < 50.0
        
        # === HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean reversion long signal
        crsi_overbought = crsi[i] > 85.0  # Strong mean reversion short signal
        crsi_moderate_low = crsi[i] < 30.0  # Moderate long signal
        crsi_moderate_high = crsi[i] > 70.0  # Moderate short signal
        
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
        if price_above_hma_1d:  # HTF bullish bias required for longs
            if is_choppy:
                # Mean reversion in range - use CRSI extremes
                if crsi_oversold:
                    desired_signal = MAX_SIZE  # High confidence
                elif crsi_moderate_low:
                    desired_signal = position_size
            else:
                # Trending market - pullback entries only
                if crsi_moderate_low:
                    desired_signal = position_size
        
        # SHORT SETUP
        if price_below_hma_1d:  # HTF bearish bias required for shorts
            if is_choppy:
                # Mean reversion in range - use CRSI extremes
                if crsi_overbought:
                    desired_signal = -MAX_SIZE  # High confidence
                elif crsi_moderate_high:
                    desired_signal = -position_size
            else:
                # Trending market - rally entries only
                if crsi_moderate_high:
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        # Exit long when CRSI becomes overbought
        if in_position and position_side > 0 and crsi[i] > 75.0:
            desired_signal = 0.0
        
        # Exit short when CRSI becomes oversold
        if in_position and position_side < 0 and crsi[i] < 25.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d:
                # Hold long if HTF still bullish
                if crsi[i] < 60.0:  # Don't hold if CRSI getting overbought
                    desired_signal = position_size
            elif position_side < 0 and price_below_hma_1d:
                # Hold short if HTF still bearish
                if crsi[i] > 40.0:  # Don't hold if CRSI getting oversold
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