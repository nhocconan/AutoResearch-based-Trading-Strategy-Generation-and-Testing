#!/usr/bin/env python3
"""
Experiment #1626: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Daily timeframe with weekly trend bias provides optimal signal quality 
for crypto perpetuals. Connors RSI (CRSI) has proven 75% win rate in mean reversion.
Combined with Choppiness Index regime detection, this adapts to market state.

Key design choices based on failure analysis:
1. CRSI thresholds: <15 for long, >85 for short (loose enough for trades)
2. CHOP regime: <40 trend, >60 range (clear separation)
3. 1w HMA for trend bias (slower = more reliable)
4. Discrete signal sizes: 0.25 base, 0.30 strong
5. 2.5x ATR trailing stoploss via signal→0
6. NO volume filter (too restrictive based on failures)

Entry logic (LOOSE to guarantee ≥30 trades/train):
- TREND (CHOP<40): 1w HMA bias + price pullback to HMA(21) + CRSI confirmation
- RANGE (CHOP>60): CRSI extremes only (mean reversion)
- NEUTRAL: 1w HMA bias + CRSI not extreme

Why this beats mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- 1d TF = fewer trades, less fee drag, higher quality signals
- Connors RSI proven superior in bear/range markets (2022-2024)
- Weekly HMA provides strong trend filter without whipsaw
- Dual regime adapts to market state

Target: Sharpe>0.6, trades≥30 train, trades≥5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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
                window = series[i - span + 1:i + 1].astype(np.float64)
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate for extremes (<10 long, >90 short)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_abs[streak_abs == 0] = 1  # avoid division issues
    rsi_streak = calculate_rsi(streak_abs, streak_period)
    
    # Component 3: Percentile rank of daily returns over last 100 days
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = np.diff(close) / close[:-1] * 100
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = returns[i - rank_period:i]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            count_below = np.sum(valid_window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(valid_window)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_3_2_100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 120  # Need 100+ for CRSI percentile rank
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi_3_2_100[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 40.0
        is_range_regime = chop > 60.0
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds for trades) ===
        crsi_val = crsi_3_2_100[i]
        crsi_oversold = crsi_val < 20  # Loose: was 10
        crsi_overbought = crsi_val > 80  # Loose: was 90
        crsi_extreme_low = crsi_val < 15
        crsi_extreme_high = crsi_val > 85
        
        # === PULLBACK TO HMA(21) ===
        pullback_long = close[i] < hma_21[i] * 1.02 and close[i] > hma_21[i] * 0.98
        pullback_short = close[i] > hma_21[i] * 0.98 and close[i] < hma_21[i] * 1.02
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else False
        donchian_breakout_short = close[i] < donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: 1w bias + pullback + CRSI confirmation
        if is_trend_regime:
            # LONG: 1w bullish + pullback to HMA21 + CRSI not overbought
            if price_above_1w and pullback_long and crsi_val < 70:
                desired_signal = SIZE_STRONG if donchian_breakout_long else SIZE_BASE
            
            # SHORT: 1w bearish + pullback to HMA21 + CRSI not oversold
            elif price_below_1w and pullback_short and crsi_val > 30:
                desired_signal = -SIZE_STRONG if donchian_breakout_short else -SIZE_BASE
        
        # RANGE REGIME: CRSI extremes only (mean reversion)
        elif is_range_regime:
            # LONG: CRSI extreme low (oversold)
            if crsi_extreme_low:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme high (overbought)
            elif crsi_extreme_high:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: 1w HMA bias + CRSI moderate extremes
        else:
            # LONG: 1w bullish + CRSI oversold
            if price_above_1w and crsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + CRSI overbought
            elif price_below_1w and crsi_overbought:
                desired_signal = -SIZE_BASE
        
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