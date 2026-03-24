#!/usr/bin/env python3
"""
Experiment #031: 4h Primary + 1d HTF — Simplified RSI Dual Regime with Volatility Filter

Hypothesis: #019 (Sharpe=0.368) outperformed #021 (Sharpe=0.253) because:
1. Regular RSI generates MORE trades than complex CRSI (critical for avoiding 0-trade failure)
2. HMA reacts faster than KAMA for trend detection
3. Simpler regime logic = fewer filters blocking valid entries

Key changes from #021:
- RSI(14) instead of CRSI - simpler, more trades, proven in #019
- HMA(21) instead of KAMA - faster trend response
- Relaxed CHOP thresholds: >50 choppy, <40 trending (vs 55/45)
- RSI thresholds: 25/75 instead of 15/85 (more entry opportunities)
- Volatility expansion filter: ATR(7)/ATR(21) > 1.2 confirms breakout validity
- Asymmetric sizing: 0.30 with HTF alignment, 0.20 counter-trend

Entry Logic:
- CHOPPY (CHOP>50): RSI<25 long, RSI>75 short (mean reversion)
- TRENDING (CHOP<40): HMA slope + 1d bias confirmation + vol expansion
- Neutral (40-50): Only trade WITH 1d trend, reduced size
- Funding contrarian overlay: +0.05 when funding<-0.005%, -0.05 when funding>0.005%
- Size: 0.30 with 1d alignment, 0.20 without

Risk: 2.5x ATR trailing stop, max signal 0.35, discrete levels
Target: Sharpe>0.40, trades>40/symbol train, >5/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_hma_chop_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
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
    """Choppiness Index - regime detection (100=choppy, 0=trending)"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi = calculate_rsi(close, period=14)
    hma_4h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    atr_fast = calculate_atr(high, low, close, period=7)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Try to load funding data
    funding_data = None
    try:
        funding_data = load_funding_data("BTCUSDT")
    except Exception:
        funding_data = None
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 40.0
        
        # === HTF TREND BIAS (1d) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # HMA 4h slope for short-term trend
        hma_4h_slope = 0.0
        if i >= 3 and not np.isnan(hma_4h[i-3]):
            hma_4h_slope = (hma_4h[i] - hma_4h[i-3]) / hma_4h[i-3] if hma_4h[i-3] > 1e-10 else 0.0
        
        # === VOLATILITY EXPANSION FILTER ===
        vol_expansion = False
        if not np.isnan(atr_fast[i]) and atr[i] > 1e-10:
            vol_ratio = atr_fast[i] / atr[i]
            vol_expansion = vol_ratio > 1.15
        
        # === FUNDING RATE CONTRARIAN ===
        funding_signal = 0.0
        try:
            funding_rate = get_funding_at_time(funding_data, open_time[i])
            if funding_rate > 0.005:  # Bullish funding = contrarian short
                funding_signal = -0.05
            elif funding_rate < -0.005:  # Bearish funding = contrarian long
                funding_signal = 0.05
        except Exception:
            funding_signal = 0.0
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - RSI extremes (RELAXED for trade gen)
            # Long: RSI < 25 (oversold in chop)
            if rsi[i] < 25.0:
                if hma_1d_bull:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength + funding_signal
            
            # Short: RSI > 75 (overbought in chop)
            elif rsi[i] > 75.0:
                if hma_1d_bear:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength + funding_signal
        
        elif is_trending:
            # TREND REGIME - HMA slope + HTF bias + vol expansion
            # Long: HMA slope > 0 + 1d bullish + vol expansion
            if hma_4h_slope > 0.0005:
                if hma_1d_bull and vol_expansion:
                    signal_strength = BASE_SIZE
                elif hma_1d_bull:
                    signal_strength = REDUCED_SIZE
                elif vol_expansion:
                    signal_strength = REDUCED_SIZE * 0.7
                else:
                    signal_strength = 0.0
                desired_signal = signal_strength + funding_signal
            
            # Short: HMA slope < 0 + 1d bearish + vol expansion
            elif hma_4h_slope < -0.0005:
                if hma_1d_bear and vol_expansion:
                    signal_strength = BASE_SIZE
                elif hma_1d_bear:
                    signal_strength = REDUCED_SIZE
                elif vol_expansion:
                    signal_strength = REDUCED_SIZE * 0.7
                else:
                    signal_strength = 0.0
                desired_signal = -signal_strength + funding_signal
        
        else:
            # NEUTRAL REGIME (40 <= CHOP <= 50) - only trade WITH 1d trend
            if hma_1d_bull and hma_4h_slope > 0.0005:
                desired_signal = REDUCED_SIZE + funding_signal
            elif hma_1d_bear and hma_4h_slope < -0.0005:
                desired_signal = -REDUCED_SIZE + funding_signal
        
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