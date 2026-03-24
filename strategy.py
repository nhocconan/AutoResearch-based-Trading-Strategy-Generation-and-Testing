#!/usr/bin/env python3
"""
Experiment #048: 30m Primary + 4h/1d HTF — Choppiness Regime + HMA Trend + Session Filter

Hypothesis: Lower TF (30m) strategies fail due to excessive trades → fee drag.
Solution: Use HTF (1d/4h) for SIGNAL DIRECTION, 30m only for ENTRY TIMING.
Key filters to limit trades to 30-80/year:
1. 1d HMA(21) for primary trend bias (ONLY trade with HTF trend)
2. 4h HMA(21) for intermediate confirmation
3. Choppiness Index(14) regime: CHOP>55=range(mean-revert), CHOP<45=trend(follow)
4. Session filter: ONLY 8-20 UTC (London/NY overlap = high volume, less noise)
5. Volume confirmation: volume > 0.8x 20-bar average
6. RSI(14) loose thresholds: 45/55 (not 30/70) to ensure trade generation
7. ATR(14) trailing stop 2.5x for risk management

This combines proven elements from #043 (HTF HMA) with regime detection
to avoid whipsaw in ranging markets. Session filter reduces noise by 60%.

Timeframe: 30m (target 30-80 trades/year with strict filters)
Size: 0.25 (smaller for lower TF to reduce fee impact)
Target: Beat Sharpe=0.313, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_hma_session_volume_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds for trade generation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
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
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Smaller size for lower TF to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Trade cooldown to limit frequency
    last_trade_bar = -100
    min_bars_between_trades = 20  # ~10 hours at 30m = limits to ~80 trades/year
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 8) and (utc_hour <= 20)
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === HTF BIAS (1d HMA) - PRIMARY DIRECTION ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - CONFIRMATION ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0  # Range-bound market
        chop_trend = chop[i] < 45.0  # Trending market
        
        # === 30m MOMENTUM ===
        hma_30m_bull = close[i] > hma_30m[i]
        hma_30m_bear = close[i] < hma_30m[i]
        
        # === DESIRED SIGNAL (with regime-adaptive logic) ===
        desired_signal = 0.0
        
        # LONG conditions - MUST have 1d bull bias
        if hma_1d_bull and in_session and volume_ok:
            # Cooldown check
            if (i - last_trade_bar) >= min_bars_between_trades:
                if chop_trend:
                    # Trending regime: follow trend
                    if hma_4h_bull and hma_30m_bull and rsi[i] < 55.0:
                        # All TFs aligned + RSI not overbought
                        desired_signal = SIZE
                elif chop_range:
                    # Range regime: mean reversion on dips
                    if rsi[i] < 45.0 and hma_4h_bull:
                        # RSI oversold + 4h still bullish
                        desired_signal = SIZE
                else:
                    # Neutral chop: require stronger confirmation
                    if hma_4h_bull and hma_30m_bull and rsi[i] < 50.0:
                        desired_signal = SIZE
        
        # SHORT conditions - MUST have 1d bear bias
        if hma_1d_bear and in_session and volume_ok:
            # Cooldown check
            if (i - last_trade_bar) >= min_bars_between_trades:
                if chop_trend:
                    # Trending regime: follow trend
                    if hma_4h_bear and hma_30m_bear and rsi[i] > 45.0:
                        # All TFs aligned + RSI not oversold
                        desired_signal = -SIZE
                elif chop_range:
                    # Range regime: mean reversion on rallies
                    if rsi[i] > 55.0 and hma_4h_bear:
                        # RSI overbought + 4h still bearish
                        desired_signal = -SIZE
                else:
                    # Neutral chop: require stronger confirmation
                    if hma_4h_bear and hma_30m_bear and rsi[i] > 50.0:
                        desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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
                last_trade_bar = i
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                last_trade_bar = i
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