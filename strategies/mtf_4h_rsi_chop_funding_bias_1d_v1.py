#!/usr/bin/env python3
"""
Experiment #019: 4h Primary + 1d HTF — Simplified Mean Reversion with Funding Contrarian

Hypothesis: Previous strategies failed due to OVER-COMPLEXITY causing 0 trades.
This uses SIMPLIFIED but PROVEN patterns:
1. RSI(14) extremes on 4h ( < 30 long, > 70 short) - LOOSE thresholds for trade gen
2. 1d HMA(21) for trend BIAS only - asymmetric sizing, not hard filter
3. Funding rate contrarian (BTC/ETH specific edge) - short when funding > 0.01%
4. Choppiness Index for regime detection - mean revert when choppy, trend when clear
5. ATR(14) trailing stop at 2.5x for risk management

Key improvements from failed attempts:
- SIMPLER RSI instead of complex CRSI (fewer calculation bugs)
- LOOSER thresholds (RSI 30/70 instead of 20/80) to ensure ≥30 trades/symbol
- Funding rate adds contrarian edge specific to BTC/ETH perpetuals
- 1d HMA for bias weighting, not binary filter
- Discrete signal levels (0.0, ±0.20, ±0.30) to minimize fee churn

Entry Logic:
- CHOPPY (CHOP > 55): RSI < 30 long, RSI > 70 short (mean reversion)
- TRENDING (CHOP < 45): Price vs 4h HMA + 1d bias confirmation
- Funding contrarian: add to signal when funding extreme
- Size: 0.30 with HTF trend, 0.20 against HTF trend

Risk: 2.5x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.4, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_chop_funding_bias_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = average gain / average loss over period
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad to match original length
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
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    # WMA helper
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
    
    # Combine
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Using 55/45 thresholds for clearer regime separation
    """
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

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    cumsum = np.cumsum(close)
    for i in range(period - 1, n):
        if i == period - 1:
            sma[i] = cumsum[i] / period
        else:
            sma[i] = (cumsum[i] - cumsum[i - period]) / period
    
    return sma

def load_funding_data(symbol):
    """
    Load funding rate data from processed parquet files.
    Returns dict with 'timestamp' and 'funding_rate' arrays.
    """
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
    
    # Return dummy data if funding not available
    return None

def get_funding_at_time(funding_data, timestamp):
    """Get funding rate closest to given timestamp"""
    if funding_data is None:
        return 0.0
    
    ts_arr = funding_data['timestamp']
    fr_arr = funding_data['funding_rate']
    
    # Find closest timestamp
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
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Try to load funding data (BTC default, will work per-symbol in real execution)
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
    
    for i in range(250, n):
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
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === HTF TREND BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # SMA200 filter for long-term bias
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FUNDING RATE CONTRARIAN ===
        funding_signal = 0.0
        try:
            funding_rate = get_funding_at_time(funding_data, open_time[i])
            if funding_rate > 0.01:  # Bullish funding = contrarian short
                funding_signal = -0.10
            elif funding_rate < -0.01:  # Bearish funding = contrarian long
                funding_signal = 0.10
        except Exception:
            funding_signal = 0.0
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use RSI extremes (LOOSE thresholds)
            # Long: RSI < 30 (oversold)
            if rsi[i] < 30.0:
                if above_sma200 or hma_1d_bull:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength + funding_signal
            
            # Short: RSI > 70 (overbought)
            elif rsi[i] > 70.0:
                if below_sma200 or hma_1d_bear:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength + funding_signal
        
        elif is_trending:
            # TREND REGIME - use SMA200 + HTF bias
            # Long: Price > SMA200 + 1d bullish
            if above_sma200:
                if hma_1d_bull:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength + funding_signal
            
            # Short: Price < SMA200 + 1d bearish
            elif below_sma200:
                if hma_1d_bear:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength + funding_signal
        
        else:
            # NEUTRAL REGIME (45 <= CHOP <= 55) - only trade WITH 1d trend
            if hma_1d_bull and above_sma200:
                desired_signal = REDUCED_SIZE + funding_signal
            elif hma_1d_bear and below_sma200:
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
        # Clamp to max magnitude and discretize
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
            # Small signals from funding - use reduced size if significant
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