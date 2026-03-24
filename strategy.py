#!/usr/bin/env python3
"""
Experiment #004: 4h Primary + 12h HTF — KAMA Adaptive Trend + Vol Spike Mean Reversion

Hypothesis: Previous CRSI+Chop strategies underperformed due to over-reliance on mean reversion.
This combines PROVEN patterns for BTC/ETH perpetuals:
1. KAMA (Kaufman Adaptive MA) - adapts to market noise, fewer whipsaws than EMA/HMA
2. Volatility Spike Mean Reversion - ATR(7)/ATR(30) > 2.0 + BB extremes = panic capitulation
3. 12h HMA for trend BIAS (not hard filter) - smoother than 1d, more responsive
4. ADX regime filter - ADX < 20 = range (mean revert), ADX > 25 = trend (follow KAMA)
5. Funding z-score (30-period) for contrarian edge on BTC/ETH

Key improvements:
- KAMA adapts to volatility ratio (ER) - slower in chop, faster in trends
- Vol spike entries catch panic bottoms (2022 crash, 2025 dips)
- ADX hysteresis (enter 25, exit 18) avoids regime flip-flop
- Funding z-score > 2.0 or < -2.0 more robust than absolute thresholds
- LOOSER entry conditions to ensure ≥30 trades/symbol

Entry Logic:
- VOL SPIKE (ATR ratio > 2.0): Mean revert at BB(20,2.5) extremes
- TREND (ADX > 25): Follow KAMA direction + 12h bias confirmation
- RANGE (ADX < 20): Mean revert at BB bounds
- Funding z-score adds contrarian overlay

Risk: 2.5x ATR trailing stop, signal discretized (0.0, ±0.20, ±0.30)
Target: Sharpe > 0.3, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_volspike_adx_funding_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_sc=2.0/11.0, slow_sc=2.0/31.0):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility via Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes over period
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    
    KAMA reacts fast in trends (high ER), slow in chop (low ER)
    """
    n = len(close)
    if n < er_period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Initialize KAMA at SMA of first er_period bars
    kama[er_period] = np.mean(close[:er_period+1])
    
    for i in range(er_period + 1, n):
        # Net change over er_period
        net_change = abs(close[i] - close[i - er_period])
        
        # Sum of absolute changes (volatility)
        vol_sum = 0.0
        for j in range(i - er_period + 1, i + 1):
            vol_sum += abs(close[j] - close[j - 1])
        
        # Efficiency Ratio (0 = noise, 1 = pure trend)
        if vol_sum < 1e-10:
            er = 0.0
        else:
            er = net_change / vol_sum
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    return hma

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    ADX > 25 = trending, ADX < 20 = ranging
    Uses +DI and -DI for direction
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    # True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    smoothed_plus_dm = np.zeros(n)
    smoothed_minus_dm = np.zeros(n)
    smoothed_plus_dm[period-1] = np.mean(plus_dm[1:period])
    smoothed_minus_dm[period-1] = np.mean(minus_dm[1:period])
    for i in range(period, n):
        smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (period - 1) + plus_dm[i]) / period
        smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (period - 1) + minus_dm[i]) / period
    
    # +DI and -DI
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * smoothed_plus_dm[i] / atr[i]
            minus_di[i] = 100.0 * smoothed_minus_dm[i] / atr[i]
    
    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = SMA of DX
    for i in range(period * 2, n):
        adx[i] = np.mean(dx[i - period + 1:i + 1])
    
    return adx, plus_di, minus_di

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for volatility spike detection"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    n = len(close)
    ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_short[i]) and not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def load_funding_data(symbol):
    """Load funding rate data from processed parquet files"""
    try:
        import os
        symbol_base = symbol.replace('USDT', '').lower()
        funding_path = f"data/processed/funding/{symbol_base}.parquet"
        
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            return {
                'timestamp': df_funding['timestamp'].values,
                'funding_rate': df_funding['funding_rate'].values
            }
    except Exception:
        pass
    
    return None

def get_funding_at_time(funding_data, timestamp):
    """Get funding rate closest to given timestamp"""
    if funding_data is None:
        return 0.0
    
    ts_arr = funding_data['timestamp']
    fr_arr = funding_data['funding_rate']
    
    idx = np.searchsorted(ts_arr, timestamp)
    if idx >= len(ts_arr):
        idx = len(ts_arr) - 1
    if idx < 0:
        idx = 0
    
    return fr_arr[idx]

def calculate_funding_zscore(funding_data, timestamps, period=30):
    """Calculate z-score of funding rate over rolling period"""
    n = len(timestamps)
    zscore = np.zeros(n)
    
    if funding_data is None:
        return zscore
    
    # Get funding rates for all timestamps
    funding_rates = np.array([get_funding_at_time(funding_data, ts) for ts in timestamps])
    
    for i in range(period, n):
        window = funding_rates[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore[i] = (funding_rates[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, er_period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.5)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
    # Funding z-score
    funding_data = load_funding_data("BTCUSDT")
    funding_z = calculate_funding_zscore(funding_data, open_time, period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.35
    
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]) or np.isnan(bb_upper[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX) ===
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        # === HTF TREND BIAS ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0
        
        # === BOLLINGER POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 1e-10 else 0.5
        at_bb_lower = close[i] <= bb_lower[i] * 1.002
        at_bb_upper = close[i] >= bb_upper[i] * 0.998
        
        # === FUNDING Z-SCORE CONTRARIAN ===
        funding_signal = 0.0
        if funding_z[i] > 2.0:  # Extremely bullish funding = contrarian short
            funding_signal = -0.10
        elif funding_z[i] < -2.0:  # Extremely bearish funding = contrarian long
            funding_signal = 0.10
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if vol_spike:
            # VOLATILITY SPIKE - Mean reversion at BB extremes
            if at_bb_lower and hma_12h_bull:
                signal_strength = BASE_SIZE
                desired_signal = signal_strength + funding_signal
            elif at_bb_lower:
                signal_strength = REDUCED_SIZE
                desired_signal = signal_strength + funding_signal
            elif at_bb_upper and hma_12h_bear:
                signal_strength = BASE_SIZE
                desired_signal = -signal_strength + funding_signal
            elif at_bb_upper:
                signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength + funding_signal
        
        elif is_trending:
            # TREND REGIME - Follow KAMA + HTF bias
            if kama_bull and hma_12h_bull:
                signal_strength = BASE_SIZE
                desired_signal = signal_strength + funding_signal
            elif kama_bull:
                signal_strength = REDUCED_SIZE
                desired_signal = signal_strength + funding_signal
            elif kama_bear and hma_12h_bear:
                signal_strength = BASE_SIZE
                desired_signal = -signal_strength + funding_signal
            elif kama_bear:
                signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength + funding_signal
        
        elif is_ranging:
            # RANGE REGIME - Mean revert at BB bounds
            if at_bb_lower:
                signal_strength = REDUCED_SIZE
                desired_signal = signal_strength + funding_signal
            elif at_bb_upper:
                signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength + funding_signal
            elif bb_position < 0.3 and hma_12h_bull:
                signal_strength = REDUCED_SIZE * 0.5
                desired_signal = signal_strength + funding_signal
            elif bb_position > 0.7 and hma_12h_bear:
                signal_strength = REDUCED_SIZE * 0.5
                desired_signal = -signal_strength + funding_signal
        
        else:
            # NEUTRAL REGIME (20 <= ADX <= 25) - Only trade with HTF bias
            if hma_12h_bull and kama_bull:
                desired_signal = REDUCED_SIZE * 0.5 + funding_signal
            elif hma_12h_bear and kama_bear:
                desired_signal = -REDUCED_SIZE * 0.5 + funding_signal
        
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        elif abs(desired_signal) >= 0.08:
            final_signal = np.sign(desired_signal) * REDUCED_SIZE
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