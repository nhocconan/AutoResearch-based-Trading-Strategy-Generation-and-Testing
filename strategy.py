#!/usr/bin/env python3
"""
Experiment #1393: 1d Primary + 1w HTF — Dual Regime Donchian Breakout with HMA Trend

Hypothesis: The current best (mtf_1d_donchian_hma_rsi_1w_atr_v1, Sharpe=0.618) uses clean
trend-following. This experiment adds a dual-regime switch: trend-follow in trending
markets (low Choppiness), mean-reversion in choppy markets (high Choppiness).

Key insight from research: Choppiness Index > 61.8 = range (mean revert), CHOP < 38.2 = trend.
This allows the strategy to adapt to market conditions instead of forcing one approach.

Design:
1. 1w HMA(21) = ultra-long trend bias (macro filter)
2. Choppiness Index(14) = regime detection (trend vs range)
3. 1d Donchian(20/55) breakout = trend-following entries (when CHOP < 50)
4. Connors RSI = mean-reversion entries (when CHOP > 50)
5. RSI(14) momentum confirmation (wide bands 30-70)
6. ATR(14) trailing stop 2.5x = risk management
7. Position size 0.30 = conservative for daily volatility
8. Multiple entry paths per regime = ensures >=30 trades/train

Target: 20-40 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_donchian_crsi_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range-bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / (log10(n))
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            choppiness[i] = 100.0 * (sum_atr / price_range) / np.log10(period)
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean-reversion signals
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_delta = np.diff(streak_abs, prepend=streak_abs[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_smooth > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask] / streak_loss_smooth[mask]))
    rsi_streak[streak_loss_smooth <= 1e-10] = 100.0
    
    # Percent Rank component
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i-rank_period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    donchian_55_upper, donchian_55_lower = calculate_donchian(high, low, period=55)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(donchian_55_upper[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1w HMA) - primary filter ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = choppy/range (mean revert), CHOP < 50 = trending
        is_trending = choppiness[i] < 50.0
        is_choppy = choppiness[i] >= 50.0
        
        # === RSI MOMENTUM (WIDE bands to ensure trades) ===
        rsi_bull = rsi[i] > 35.0
        rsi_bear = rsi[i] < 65.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === CONNORS RSI (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_20_long = close[i] > donchian_20_upper[i-1]
        breakout_20_short = close[i] < donchian_20_lower[i-1]
        breakout_55_long = close[i] > donchian_55_upper[i-1]
        breakout_55_short = close[i] < donchian_55_lower[i-1]
        
        # === DESIRED SIGNAL - DUAL REGIME APPROACH ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND-FOLLOWING REGIME (CHOP < 50)
            # Long entries in trending regime
            if macro_bull:
                # Path 1: Donchian-20 breakout + RSI momentum
                if breakout_20_long and rsi_bull:
                    desired_signal = BASE_SIZE
                # Path 2: Donchian-55 breakout (strong trend)
                elif breakout_55_long and rsi_strong_bull:
                    desired_signal = BASE_SIZE
                # Path 3: Price above 1w HMA + RSI confirmation
                elif rsi_strong_bull:
                    desired_signal = BASE_SIZE * 0.5
            
            # Short entries in trending regime
            elif macro_bear:
                # Path 1: Donchian-20 breakout + RSI momentum
                if breakout_20_short and rsi_bear:
                    desired_signal = -BASE_SIZE
                # Path 2: Donchian-55 breakout (strong trend)
                elif breakout_55_short and rsi_strong_bear:
                    desired_signal = -BASE_SIZE
                # Path 3: Price below 1w HMA + RSI confirmation
                elif rsi_strong_bear:
                    desired_signal = -BASE_SIZE * 0.5
        
        else:
            # MEAN-REVERSION REGIME (CHOP >= 50)
            # Long entries in choppy regime (oversold)
            if macro_bull or not macro_bear:
                # Path 1: CRSI extreme oversold + 1w HMA support
                if crsi_extreme_oversold and close[i] > hma_1w_aligned[i] * 0.95:
                    desired_signal = BASE_SIZE
                # Path 2: CRSI oversold + RSI confirmation
                elif crsi_oversold and rsi[i] < 40.0:
                    desired_signal = BASE_SIZE * 0.5
                # Path 3: Donchian lower bounce in range
                elif close[i] < donchian_20_lower[i-1] * 1.02 and rsi[i] < 45.0:
                    desired_signal = BASE_SIZE * 0.5
            
            # Short entries in choppy regime (overbought)
            elif macro_bear:
                # Path 1: CRSI extreme overbought + 1w HMA resistance
                if crsi_extreme_overbought and close[i] < hma_1w_aligned[i] * 1.05:
                    desired_signal = -BASE_SIZE
                # Path 2: CRSI overbought + RSI confirmation
                elif crsi_overbought and rsi[i] > 60.0:
                    desired_signal = -BASE_SIZE * 0.5
                # Path 3: Donchian upper rejection in range
                elif close[i] > donchian_20_upper[i-1] * 0.98 and rsi[i] > 55.0:
                    desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
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