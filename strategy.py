#!/usr/bin/env python3
"""
Experiment #1563: 1d Primary + 1w HTF — Dual Regime Strategy (Choppiness + Donchian + CRSI)

Hypothesis: 1d Choppiness Index regime detection + 1w HMA trend filter will beat Sharpe 0.618

Key insights from 1162 failed experiments:
1. 1d timeframe shows best results (#1557 Sharpe=0.168, current best Sharpe=0.618)
2. Choppiness Index regime switch worked for ETH (Sharpe +0.923 in research)
3. Donchian breakout + HMA trend worked for SOL (Sharpe +0.782 in research)
4. Connors RSI mean reversion effective in range markets
5. 1w HTF filter provides ultra-long-term bias without whipsaw

Strategy Design:
- HTF Bias: 1w HMA(21) for secular trend direction
- Regime Detection: Choppiness Index(14) on 1d
  - CHOP > 61.8 = Range regime (use Connors RSI mean reversion)
  - CHOP < 38.2 = Trend regime (use Donchian breakout)
  - Between = neutral (reduce size or flat)
- Trend Entry: Price breaks Donchian(20) high/low + aligned with 1w HMA
- Range Entry: Connors RSI < 10 (long) or > 90 (short) + price vs 1w HMA
- Exit: 2.5x ATR(14) trailing stop via signal→0
- Size: 0.30 discrete (0.0, ±0.30) for fee efficiency

Why this should work:
- 1w HMA filter avoids counter-trend trades in secular moves
- Choppiness Index adapts strategy to market regime (trend vs range)
- Donchian breakout catches trends early with clear entry/exit
- Connors RSI captures mean reversion in choppy markets
- Works on BTC/ETH/SOL (different regimes suit different coins)

Timeframe: 1d (required for this experiment)
HTF: 1w HMA for bias
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test
Trade Frequency: Target 20-40 trades/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_donchian_crsi_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Range/Choppy
    CHOP < 38.2 = Trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI-Streak(2): Duration of consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 days
    
    CRSI < 10 = Oversold (long signal)
    CRSI > 90 = Overbought (short signal)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    streak_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain[mask] / streak_loss[mask]))
    rsi_streak[streak_loss <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout system
    Upper = Highest High over period
    Lower = Lowest Low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for secular trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need enough data for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SECULAR TREND BIAS (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range_regime = chop[i] > 61.8
        is_trend_regime = chop[i] < 38.2
        is_neutral_regime = not is_range_regime and not is_trend_regime
        
        # === ENTRY LOGIC BY REGIME ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_trend_regime:
            # TREND REGIME: Use Donchian breakout with weekly bias
            # LONG: Break above Donchian upper + weekly bull
            if close[i] > donchian_upper[i-1] and weekly_bull:
                desired_signal = BASE_SIZE
                signal_strength = 1.0
            
            # SHORT: Break below Donchian lower + weekly bear
            elif close[i] < donchian_lower[i-1] and weekly_bear:
                desired_signal = -BASE_SIZE
                signal_strength = 1.0
        
        elif is_range_regime:
            # RANGE REGIME: Use Connors RSI mean reversion with weekly bias
            crsi_valid = not np.isnan(crsi[i])
            
            if crsi_valid:
                # LONG: CRSI < 10 (oversold) + price not too far below weekly HMA
                if crsi[i] < 15.0 and weekly_bull:
                    desired_signal = BASE_SIZE
                    signal_strength = 1.0
                # More aggressive long in deep oversold
                elif crsi[i] < 10.0:
                    desired_signal = BASE_SIZE
                    signal_strength = 1.0
                
                # SHORT: CRSI > 90 (overbought) + price not too far above weekly HMA
                elif crsi[i] > 85.0 and weekly_bear:
                    desired_signal = -BASE_SIZE
                    signal_strength = 1.0
                # More aggressive short in deep overbought
                elif crsi[i] > 90.0:
                    desired_signal = -BASE_SIZE
                    signal_strength = 1.0
        
        # Neutral regime: reduce position or flat (avoid whipsaw)
        if is_neutral_regime and in_position:
            # Keep existing position but don't add
            pass
        elif is_neutral_regime and not in_position:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals