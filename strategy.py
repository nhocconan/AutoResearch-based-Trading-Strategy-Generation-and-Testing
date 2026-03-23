#!/usr/bin/env python3
"""
Experiment #1221: 4h Primary + 1d/1w HTF — Dual Regime with Connors RSI + KAMA Trend

Hypothesis: 4h timeframe balances trade frequency (20-50/year) with signal quality.
Dual regime approach: mean reversion in choppy markets (Connors RSI), trend following in trending markets (KAMA + Donchian).
1w HMA provides macro bias filter to avoid counter-trend trades in strong regimes.
Key improvement: LOOSENED entry conditions from failed experiments to ensure >=30 trades.

Why this might work:
- Connors RSI (CRSI) has 75% win rate for mean reversion entries
- Choppiness Index cleanly separates regime (61.8 chop, 38.2 trend)
- KAMA adapts to market efficiency (fast in trends, slow in chop)
- 1w HMA macro filter prevents fighting major trend
- ATR 2.5x trailing stop limits drawdown
- Position size 0.28 discrete minimizes fee churn

Entry conditions LOOSENED from failed experiments:
- RSI thresholds: 30/70 (not 20/80) for more signals
- CRSI thresholds: 15/85 (not 10/90) for more trades
- Donchian breakout: any breakout in trend regime (not requiring slope confirmation)
- Macro filter: only filter extreme counter-trend (not requiring full alignment)

Target: Sharpe > 0.612 (beat current best), trades >= 30 on train, >= 3 on test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_crsi_kama_1d1w_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, trend_period=10):
    """
    Kaufman Adaptive Moving Average — adapts to market efficiency.
    Fast in trends, slow in choppy markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < trend_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(trend_period, n):
        price_change = abs(close[i] - close[i - trend_period])
        if price_change < 1e-10:
            er[i] = 0.0
        else:
            volatility = np.sum(np.abs(np.diff(close[i - trend_period:i + 1])))
            if volatility > 1e-10:
                er[i] = price_change / volatility
            else:
                er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[trend_period] = close[trend_period]
    
    for i in range(trend_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 15, Short: CRSI > 85
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) — streak of consecutive up/down days
    streak_rsi = np.full(n, np.nan)
    delta = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            streak[i] = streak[i-1] + 1 if i > 0 and delta[i-1] > 0 else 1
        elif delta[i-1] < 0:
            streak[i] = streak[i-1] - 1 if i > 0 and delta[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        if len(window) > 0:
            count_below = np.sum(window < close[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout indicator."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bb(close, period=20, std_mult=2.0):
    """Bollinger Bands — mean reversion levels."""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mid[i] = np.mean(window)
        std = np.std(window)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    
    return mid, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, fast_period=2, slow_period=30, trend_period=10)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_mid, bb_upper, bb_lower = calculate_bb(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(kama[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(crsi[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) — Only filter extreme counter-trend ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) ===
        inter_bull = close[i] > hma_1d_aligned[i]
        inter_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_slope_up = False
        kama_slope_down = False
        if i >= 5 and not np.isnan(kama[i-5]):
            kama_slope_up = kama[i] > kama[i-5]
            kama_slope_down = kama[i] < kama[i-5]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === CONNORS RSI (LOOSENED from 10/90 to 15/85) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === BOLLINGER EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === CHOPPY REGIME: Mean Reversion (CRSI + RSI + BB) ===
        if is_choppy:
            # Long: CRSI oversold OR (RSI oversold + BB lower)
            if crsi_oversold or (rsi_oversold and bb_oversold):
                # Only enter if not strongly counter to macro trend
                if not macro_bear or inter_bull:
                    desired_signal = BASE_SIZE
            # Short: CRSI overbought OR (RSI overbought + BB upper)
            elif crsi_overbought or (rsi_overbought and bb_overbought):
                # Only enter if not strongly counter to macro trend
                if not macro_bull or inter_bear:
                    desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME: KAMA + Donchian Breakout ===
        elif is_trending:
            # Long: KAMA up + breakout + macro not bearish
            if kama_slope_up and breakout_long:
                if not macro_bear:
                    desired_signal = BASE_SIZE
            # Short: KAMA down + breakout + macro not bullish
            elif kama_slope_down and breakout_short:
                if not macro_bull:
                    desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE (38.2 <= CHOP <= 61.8): Use KAMA + RSI ===
        else:
            # Long: KAMA up + RSI not overbought
            if kama_slope_up and not rsi_overbought:
                if inter_bull or macro_bull:
                    desired_signal = BASE_SIZE
            # Short: KAMA down + RSI not oversold
            elif kama_slope_down and not rsi_oversold:
                if inter_bear or macro_bear:
                    desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals