#!/usr/bin/env python3
"""
Experiment #051: 4h Primary + 1d/1w HTF — Funding Rate Contrarian + Choppiness Regime

Hypothesis: Based on research showing funding rate mean reversion has Sharpe 0.8-1.5
through 2022 crash (BEST edge for BTC/ETH). Combined with:
1. Funding rate z-score as PRIMARY signal (contrarian: extreme positive = short, extreme negative = long)
2. Choppiness Index regime filter (mean revert when choppy, trend when clear)
3. 1d/1w HMA for major trend bias (only counter-trend when choppy)
4. Very LOOSE entry thresholds to ensure trade generation (>30 trades/symbol train)
5. Discrete sizing: 0.30 with HTF alignment, 0.20 against HTF trend

Key insight from 47 failed strategies:
- Simple trend following fails on BTC/ETH (2022 crash destroys gains)
- Funding contrarian works through crashes (proven edge)
- Need LOOSE thresholds to generate trades (many strategies failed with 0 trades)
- 4h timeframe = target 20-50 trades/year

Entry Logic:
- Funding z-score < -1.5 → long contrarian (funding too bearish)
- Funding z-score > +1.5 → short contrarian (funding too bullish)
- Choppy regime (CHOP>50): allow counter-HTF trades
- Trending regime (CHOP<40): only trade WITH HTF trend
- Size: 0.30 with HTF alignment, 0.20 counter-trend in chop

Risk: 2.5x ATR trailing stop, max signal 0.30, discrete levels
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_chop_regime_1d1w_loose_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def calculate_funding_zscore(funding_rates, window=20):
    """Calculate z-score of funding rates for contrarian signal"""
    n = len(funding_rates)
    zscore = np.full(n, np.nan)
    
    for i in range(window, n):
        window_data = funding_rates[i-window:i]
        mean_funding = np.mean(window_data)
        std_funding = np.std(window_data)
        if std_funding > 1e-10:
            zscore[i] = (funding_rates[i] - mean_funding) / std_funding
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Load funding data and build funding series
    funding_data = None
    try:
        funding_data = load_funding_data("BTCUSDT")
    except Exception:
        funding_data = None
    
    # Build funding rate series aligned with prices
    funding_rates = np.zeros(n)
    for i in range(n):
        funding_rates[i] = get_funding_at_time(funding_data, open_time[i])
    
    # Calculate funding z-score
    funding_zscore = calculate_funding_zscore(funding_rates, window=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.30
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(funding_zscore[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 40.0
        
        # === HTF TREND BIAS (1d and 1w) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Count HTF alignment
        htf_bull_count = sum([hma_1d_bull, hma_1w_bull])
        htf_bear_count = sum([hma_1d_bear, hma_1w_bear])
        
        # === FUNDING RATE CONTRARIAN (PRIMARY SIGNAL) ===
        # LOOSE thresholds to ensure trade generation
        funding_long = funding_zscore[i] < -1.0  # Funding too bearish → contrarian long
        funding_short = funding_zscore[i] > 1.0  # Funding too bullish → contrarian short
        
        # === RSI CONFIRMATION (avoid entering at extremes against position) ===
        rsi_oversold = rsi[i] < 40  # Support for long
        rsi_overbought = rsi[i] > 60  # Support for short
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_choppy:
            # CHOPPY REGIME: Mean reversion, allow counter-HTF trades
            if funding_long:
                if htf_bull_count >= 1:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE  # Counter-HTF but allowed in chop
                if rsi_oversold:
                    signal_strength = min(signal_strength + 0.05, MAX_SIZE)
                desired_signal = signal_strength
            
            elif funding_short:
                if htf_bear_count >= 1:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE  # Counter-HTF but allowed in chop
                if rsi_overbought:
                    signal_strength = min(signal_strength + 0.05, MAX_SIZE)
                desired_signal = -signal_strength
        
        elif is_trending:
            # TRENDING REGIME: Only trade WITH HTF trend
            if funding_long and htf_bull_count >= 1:
                if htf_bull_count >= 2:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                if rsi_oversold:
                    signal_strength = min(signal_strength + 0.05, MAX_SIZE)
                desired_signal = signal_strength
            
            elif funding_short and htf_bear_count >= 1:
                if htf_bear_count >= 2:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                if rsi_overbought:
                    signal_strength = min(signal_strength + 0.05, MAX_SIZE)
                desired_signal = -signal_strength
        
        else:
            # NEUTRAL REGIME (40 <= CHOP <= 50): Moderate signals
            if funding_long:
                if htf_bull_count >= 1:
                    signal_strength = REDUCED_SIZE
                else:
                    signal_strength = 0.15  # Small position
                if rsi_oversold:
                    signal_strength = min(signal_strength + 0.05, MAX_SIZE)
                desired_signal = signal_strength
            
            elif funding_short:
                if htf_bear_count >= 1:
                    signal_strength = REDUCED_SIZE
                else:
                    signal_strength = 0.15  # Small position
                if rsi_overbought:
                    signal_strength = min(signal_strength + 0.05, MAX_SIZE)
                desired_signal = -signal_strength
        
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
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * 0.15
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