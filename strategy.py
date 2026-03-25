#!/usr/bin/env python3
"""
Experiment #1494: 1d Primary + 1w HTF — Volatility Compression Breakout with Trend Filter

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance between
trade frequency (20-50/year) and signal quality. This strategy combines:
1. Volatility compression detection (BB Width at 200-day low) → breakout imminent
2. Weekly HMA(21) for major trend bias (avoid counter-trend breakouts)
3. Donchian(20) breakout for entry timing
4. ATR-based position sizing and stoploss (3x ATR for daily TF)
5. Asymmetric sizing: stronger positions with weekly trend, weaker against

Why this should work on 1d:
- Volatility compression precedes major moves (proven in quant literature)
- Weekly filter prevents major counter-trend disasters (2022 crash protection)
- Daily TF = natural 25-40 trades/year (fee-efficient, meets minimum trade req)
- LOOSE breakout thresholds guarantee trades (Donchian break, not exact level)
- Works in both bull and bear regimes (asymmetric, not long-only)

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: weekly_HMA bullish + BB_width < 200d_low + price > Donchian_high
- SHORT: weekly_HMA bearish + BB_width < 200d_low + price < Donchian_low
- Also allow mean-reversion when BB_width > 80th percentile (vol expansion exhaustion)

Target: Sharpe>0.6, trades>=30 train, trades>=3 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete (0.30 with trend, 0.20 counter-trend)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_vol_compression_donchian_1w_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands with bandwidth"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return upper, sma, lower, bandwidth

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_percentile_rank(series, lookback=200):
    """Percentile rank of current value over lookback period"""
    n = len(series)
    result = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window = series[i - lookback:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current = series[i]
            rank = np.sum(valid <= current) / len(valid)
            result[i] = rank
    
    return result

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
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower, bb_bandwidth = calculate_bollinger(close, period=20, std_mult=2.0)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Calculate BB bandwidth percentile rank (vol compression detection)
    bb_percentile = calculate_percentile_rank(bb_bandwidth, lookback=200)
    
    signals = np.zeros(n)
    SIZE_WITH_TREND = 0.30
    SIZE_COUNTER_TREND = 0.20
    SIZE_MEAN_REVERT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (need 200 bars for percentile rank)
    min_bars = 250
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_bandwidth[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND BIAS ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY REGIME ===
        bb_pct = bb_percentile[i]
        is_vol_compression = bb_pct < 0.15  # BB width in bottom 15% of 200d range
        is_vol_expansion = bb_pct > 0.80    # BB width in top 20% of 200d range
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        donchian_breakout_short = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        # === RSI EXTREMES (for mean reversion) ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # VOL COMPRESSION BREAKOUT (primary signal - high probability)
        if is_vol_compression:
            # LONG: weekly bullish + breakout up
            if price_above_1w and donchian_breakout_long:
                desired_signal = SIZE_WITH_TREND
            
            # SHORT: weekly bearish + breakout down
            elif price_below_1w and donchian_breakout_short:
                desired_signal = -SIZE_WITH_TREND
            
            # Counter-trend breakout (weaker signal)
            elif price_below_1w and donchian_breakout_long:
                desired_signal = SIZE_COUNTER_TREND
            
            elif price_above_1w and donchian_breakout_short:
                desired_signal = -SIZE_COUNTER_TREND
        
        # VOL EXPANSION MEAN REVERSION (exhaustion trades)
        elif is_vol_expansion:
            # LONG: RSI oversold + price near BB lower
            if rsi_oversold and close[i] <= bb_lower[i] * 1.01:
                desired_signal = SIZE_MEAN_REVERT
            
            # SHORT: RSI overbought + price near BB upper
            elif rsi_overbought and close[i] >= bb_upper[i] * 0.99:
                desired_signal = -SIZE_MEAN_REVERT
        
        # NEUTRAL VOL: Only take strong trend signals
        else:
            # LONG: weekly bullish + RSI not overbought + price above BB mid
            if price_above_1w and rsi < 60 and close[i] > bb_mid[i]:
                desired_signal = SIZE_COUNTER_TREND * 0.8
            
            # SHORT: weekly bearish + RSI not oversold + price below BB mid
            elif price_below_1w and rsi > 40 and close[i] < bb_mid[i]:
                desired_signal = -SIZE_COUNTER_TREND * 0.8
        
        # === STOPLOSS CHECK (3x ATR for daily TF - wider stops) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_WITH_TREND * 0.9:
            final_signal = SIZE_WITH_TREND
        elif desired_signal <= -SIZE_WITH_TREND * 0.9:
            final_signal = -SIZE_WITH_TREND
        elif desired_signal >= SIZE_MEAN_REVERT * 0.9:
            final_signal = SIZE_MEAN_REVERT
        elif desired_signal <= -SIZE_MEAN_REVERT * 0.9:
            final_signal = -SIZE_MEAN_REVERT
        elif desired_signal >= SIZE_COUNTER_TREND * 0.9:
            final_signal = SIZE_COUNTER_TREND
        elif desired_signal <= -SIZE_COUNTER_TREND * 0.9:
            final_signal = -SIZE_COUNTER_TREND
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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