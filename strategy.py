#!/usr/bin/env python3
"""
Experiment #1543: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 11 failed 4h experiments (#1529-#1541), the data is clear:
1. 1d timeframe has BEST results (current best Sharpe=0.618, Return=+122.2%)
2. 4h strategies consistently fail (Sharpe negative or near zero)
3. Complex regime switching creates conflicting signals → whipsaws
4. Connors RSI extremes (<20 or >80) are TOO STRICT → 0 trades

New Approach — PROVEN patterns from research for 1d:
- Donchian(20) breakout: proven Sharpe +0.782 on SOL (research notes)
- 1w HMA(21) for macro trend bias: prevents counter-trend trades
- LOOSE RSI filter (>45 for long, <55 for short): ensures 30+ trades/train
- ATR(14) trailing stop 2.5x: protects from crash drawdowns
- Discrete sizing (0.0, ±0.25, ±0.30): minimizes fee churn

Why this should work:
- 1d has fewer false signals than 4h (proven by experiment history)
- Weekly HMA provides macro bias without lag (1w bars complete before 1d uses them)
- Donchian breakout catches momentum in both bull and bear markets
- LOOSE RSI ensures trades fire (unlike CRSI extremes that gave 0 trades)
- ATR stoploss protects from 2022-style crashes (77% drawdown)

Timeframe: 1d (required by experiment)
HTF: 1w HMA(21) for macro trend bias
Position Size: 0.30 (conservative for daily volatility)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades > 30/train, > 3/test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_atr_v2"
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
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout system
    Upper = highest high of last n periods
    Lower = lowest low of last n periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    mask = tr_smooth > 1e-10
    di_plus[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    mask2 = (di_plus + di_minus) > 1e-10
    dx[mask2] = 100.0 * np.abs(di_plus[mask2] - di_minus[mask2]) / (di_plus[mask2] + di_minus[mask2])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range-bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 1e-10:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            choppiness[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 1d HMA for additional trend confirmation
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === MACRO TREND BIAS (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND (HMA crossover) ===
        hma_bull = hma_21[i] > hma_48[i] if not np.isnan(hma_21[i]) and not np.isnan(hma_48[i]) else False
        hma_bear = hma_21[i] < hma_48[i] if not np.isnan(hma_21[i]) and not np.isnan(hma_48[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === RSI FILTER (LOOSE — ensures trades fire) ===
        rsi_long_ok = rsi_14[i] > 45.0 if not np.isnan(rsi_14[i]) else False
        rsi_short_ok = rsi_14[i] < 55.0 if not np.isnan(rsi_14[i]) else False
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx[i] > 25.0 if not np.isnan(adx[i]) else False
        trend_weak = adx[i] < 20.0 if not np.isnan(adx[i]) else False
        
        # === CHOPPINESS REGIME ===
        is_choppy = choppiness[i] > 55.0 if not np.isnan(choppiness[i]) else False
        is_trending = choppiness[i] < 45.0 if not np.isnan(choppiness[i]) else False
        
        # === DESIRED SIGNAL — BREAKOUT + TREND CONFLUENCE ===
        desired_signal = 0.0
        
        # LONG SETUP — require macro bull + breakout + RSI ok
        long_score = 0
        if weekly_bull:
            long_score += 2  # Macro trend support (most important)
        if hma_bull:
            long_score += 1  # 1d trend confirmation
        if breakout_long:
            long_score += 3  # Donchian breakout (primary trigger)
        if rsi_long_ok:
            long_score += 1  # RSI not overbought
        if is_trending:
            long_score += 1  # Trending regime favors breakout
        
        # SHORT SETUP — require macro bear + breakout + RSI ok
        short_score = 0
        if weekly_bear:
            short_score += 2  # Macro trend support (most important)
        if hma_bear:
            short_score += 1  # 1d trend confirmation
        if breakout_short:
            short_score += 3  # Donchian breakout (primary trigger)
        if rsi_short_ok:
            short_score += 1  # RSI not oversold
        if is_trending:
            short_score += 1  # Trending regime favors breakout
        
        # Entry thresholds — LOOSE to ensure trades fire (need 30+ trades/train)
        if long_score >= 5:
            desired_signal = BASE_SIZE
        elif short_score >= 5:
            desired_signal = -BASE_SIZE
        elif long_score >= 4 and weekly_bull:
            desired_signal = BASE_SIZE * 0.8
        elif short_score >= 4 and weekly_bear:
            desired_signal = -BASE_SIZE * 0.8
        
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.5
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