#!/usr/bin/env python3
"""
Experiment #1191: 4h Primary + 1d HTF — Funding Rate Mean Reversion + Vol Spike

Hypothesis: Research shows funding rate mean reversion has Sharpe 0.8-1.5 through 2022 crash
for BTC/ETH. This is the BEST EDGE mentioned in research. Combined with vol spike
confirmation and asymmetric regime filtering, this should beat the current best (Sharpe=0.612).

Key components:
1. Funding Rate Z-score (30-day): Z > +2 = crowded longs → short, Z < -2 = crowded shorts → long
2. Vol Spike Confirmation: ATR(7)/ATR(30) > 1.8 confirms panic/extreme conditions
3. Asymmetric Regime: 1d HMA(50) filters direction (only long above, only short below)
4. BB Extreme: Price must touch BB(20,2.5) bands for mean reversion entries
5. Conservative sizing: 0.25-0.30 discrete, 2.5x ATR trailing stop

Target: 30-50 trades/year, Sharpe > 0.612
Position Size: 0.28 discrete
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_zscore_vol_spike_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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

def calculate_zscore(series, period=30):
    """Z-score for mean reversion detection."""
    n = len(series)
    zscore = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = series[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore[i] = (series[i] - mean) / std
    
    return zscore

def load_funding_data(symbol):
    """
    Load funding rate data from processed parquet files.
    Returns array aligned with prices length, or None if unavailable.
    """
    try:
        # Map symbol to funding file
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        funding_symbol = symbol_map.get(symbol, symbol)
        funding_path = f"data/processed/funding/{funding_symbol}.parquet"
        
        df_funding = pd.read_parquet(funding_path)
        return df_funding['funding_rate'].values
    except Exception:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_mid, bb_upper, bb_lower = calculate_bb(close, period=20, std_mult=2.5)
    
    # Vol spike ratio: ATR(7)/ATR(30)
    vol_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    # Try to load funding rate data
    symbol = prices.get('symbol', 'BTCUSDT') if isinstance(prices, pd.DataFrame) else 'BTCUSDT'
    funding_rates = load_funding_data(symbol)
    
    # If funding data available, calculate Z-score
    if funding_rates is not None and len(funding_rates) >= n:
        funding_zscore = calculate_zscore(funding_rates[:n], period=30)
        has_funding = True
    else:
        funding_zscore = np.full(n, np.nan)
        has_funding = False
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or np.isnan(vol_ratio[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === BOLLINGER EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === FUNDING RATE SIGNALS (if available) ===
        funding_short_signal = False
        funding_long_signal = False
        
        if has_funding and not np.isnan(funding_zscore[i]):
            # Z > +2 = crowded longs → short signal
            if funding_zscore[i] > 2.0:
                funding_short_signal = True
            # Z < -2 = crowded shorts → long signal
            elif funding_zscore[i] < -2.0:
                funding_long_signal = True
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === FUNDING MEAN REVERSION (primary signal if available) ===
        if has_funding:
            # Long: funding Z < -2 + vol spike + BB oversold + macro bull
            if funding_long_signal and vol_spike and bb_oversold and macro_bull:
                desired_signal = BASE_SIZE
            # Short: funding Z > +2 + vol spike + BB overbought + macro bear
            elif funding_short_signal and vol_spike and bb_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            # Fallback: just funding extreme + macro alignment (more trades)
            elif funding_long_signal and macro_bull and bb_oversold:
                desired_signal = BASE_SIZE * 0.5
            elif funding_short_signal and macro_bear and bb_overbought:
                desired_signal = -BASE_SIZE * 0.5
        
        # === VOL SPIKE REVERSION (fallback if no funding data) ===
        else:
            # Long: vol spike + BB oversold + macro bull
            if vol_spike and bb_oversold and macro_bull:
                desired_signal = BASE_SIZE
            # Short: vol spike + BB overbought + macro bear
            elif vol_spike and bb_overbought and macro_bear:
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
        if desired_signal >= BASE_SIZE * 0.9:
            desired_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.9:
            desired_signal = -BASE_SIZE
        elif desired_signal > 0:
            desired_signal = BASE_SIZE * 0.5
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
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