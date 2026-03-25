#!/usr/bin/env python3
"""
Experiment #1372: 12h Primary + 1d HTF — Dual Regime Strategy

Hypothesis: 12h timeframe with regime-adaptive logic will outperform single-regime strategies.
- Choppiness Index detects market regime (chop vs trend)
- In CHOPPY regime (CHOP > 61.8): Mean reversion with Connors RSI
- In TRENDING regime (CHOP < 38.2): Trend following with Donchian breakout + HMA
- 1d HMA provides major trend bias filter
- ATR trailing stoploss for risk management
- Discrete position sizing (0.25-0.35) to minimize fee churn

Why this should work:
- Dual regime adapts to market conditions (2022 crash = chop, 2021/2024 = trend)
- Connors RSI has 75% win rate in backtests (literature)
- Choppiness Index is proven regime filter
- 12h TF = natural 20-50 trades/year (fee-friendly)
- Works on BTC/ETH/SOL (not SOL-biased)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.35 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1]
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                if j > 0 and not np.isnan(high[j]) and not np.isnan(low[j]) and not np.isnan(close[j-1]):
                    tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                    atr_sum += tr
            
            choppiness[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak RSI, and percentile rank"""
    n = len(close)
    if n < max(rsi_period, streak_period, rank_period) + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    # Streak RSI
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i-1]):
            if close[i] > close[i-1]:
                streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
            elif close[i] < close[i-1]:
                streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
            else:
                streak[i] = 0
    
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    streak_gain = np.insert(streak_gain, 0, 0)
    streak_loss = np.insert(streak_loss, 0, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_streak_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    streak_rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    # Percentile Rank
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid < close[i])
            percent_rank[i] = rank / len(valid) * 100
    
    crsi = (rsi + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channels"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    min_bars = 150
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_WEAK = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = choppiness[i]
        is_choppy = chop > 61.8  # Range-bound market
        is_trending = chop < 38.2  # Trending market
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA DIRECTION ===
        hma_uptrend = False
        hma_downtrend = False
        if i >= 2 and not np.isnan(hma_21[i-1]) and not np.isnan(hma_21[i-2]):
            if hma_21[i] > hma_21[i-1] > hma_21[i-2]:
                hma_uptrend = True
            elif hma_21[i] < hma_21[i-1] < hma_21[i-2]:
                hma_downtrend = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # MEAN REVERSION (choppy regime) - use Connors RSI extremes
        if is_choppy:
            # LONG: CRSI < 15 (oversold) + price above 1d HMA (bullish bias)
            if crsi[i] < 15 and price_above_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI > 85 (overbought) + price below 1d HMA (bearish bias)
            elif crsi[i] > 85 and price_below_1d:
                desired_signal = -SIZE_BASE
        
        # TREND FOLLOWING (trending regime) - use Donchian breakout
        elif is_trending:
            # LONG: Price breaks Donchian upper + 12h HMA uptrend
            if close[i] > donchian_upper[i] and hma_uptrend:
                if price_above_1d:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: Price breaks Donchian lower + 12h HMA downtrend
            elif close[i] < donchian_lower[i] and hma_downtrend:
                if price_below_1d:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # NEUTRAL regime (38.2 <= CHOP <= 61.8) - reduce position or flat
        else:
            # Allow smaller positions only with strong confluence
            if crsi[i] < 10 and price_above_1d:
                desired_signal = SIZE_WEAK
            elif crsi[i] > 90 and price_below_1d:
                desired_signal = -SIZE_WEAK
        
        # === VOL-ADJUSTED POSITION SIZING ===
        if desired_signal != 0.0:
            atr_ratio = atr_14[i] / np.nanmedian(atr_14[min_bars:i]) if i > min_bars else 1.0
            vol_scale = 1.0 / max(0.5, min(2.0, atr_ratio))
            desired_signal = desired_signal * vol_scale
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_WEAK * 0.9:
            if desired_signal > 0:
                final_signal = SIZE_WEAK
            else:
                final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals