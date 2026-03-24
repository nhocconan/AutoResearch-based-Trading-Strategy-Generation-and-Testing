#!/usr/bin/env python3
"""
Experiment #006: 12h Primary + 1d HTF — Funding Mean Reversion + Vol Spike + HMA Trend

Hypothesis: Previous 12h strategies failed due to OVER-FILTERING with CRSI/Choppiness.
This uses PROVEN patterns specifically for BTC/ETH perpetuals:
1. FUNDING RATE CONTRARIAN (BEST edge for BTC/ETH): Z-score < -2 → long, > +2 → short
   Reported Sharpe 0.8-1.5 through 2022 crash in academic research
2. VOLATILITY SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20, 2.5) → long
   Captures "vol crush" after panic selloffs
3. 1d HMA for trend BIAS only (not hard filter) - asymmetric sizing
4. Donchian(20) breakout confirmation for entry timing precision
5. ATR trailing stop at 2.5x for risk management

Key improvements from failed attempts:
- Funding rate is PRIMARY signal (not secondary filter) - ensures trade generation
- Vol spike is CONFIRMATION not requirement - more trades than strict filters
- 1d HMA for bias weighting, not binary filter
- Discrete signal levels (0.0, ±0.25, ±0.30) to minimize fee churn
- LOOSE entry thresholds to ensure ≥30 trades/symbol on train

Entry Logic:
- Funding Z < -2.0: Long (crowded shorts will cover)
- Funding Z > +2.0: Short (crowded longs will liquidate)
- Vol spike (ATR7/ATR30 > 1.8) + BB extreme: Add to signal strength
- 1d HMA bias: Full size with trend, half size against trend
- Donchian break: Entry timing confirmation (price breaks 20-bar low/high)

Risk: 2.5x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.3, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_funding_volspike_hma_donchian_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with middle, upper, lower"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return middle, upper, lower

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(close := high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_zscore(series, period=30):
    """Z-score of a series over rolling period"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    zscore = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = series[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore[i] = (series[i] - mean) / std
        else:
            zscore[i] = 0.0
    return zscore

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
    
    # Calculate primary (12h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Try to load funding data for the symbol
    funding_data = None
    try:
        # Extract symbol from prices metadata if available
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0] if len(symbol) > 0 else 'BTCUSDT'
        funding_data = load_funding_data(symbol)
    except Exception:
        funding_data = None
    
    # Calculate funding Z-score over time
    funding_zscore = np.full(n, np.nan)
    if funding_data is not None:
        funding_rates = np.array([get_funding_at_time(funding_data, open_time[i]) for i in range(n)])
        funding_zscore = calculate_zscore(funding_rates, period=30)
    
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
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
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
        if np.isnan(bb_mid[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === FUNDING RATE CONTRARIAN (PRIMARY SIGNAL) ===
        funding_signal = 0.0
        funding_strength = 0.0
        
        if not np.isnan(funding_zscore[i]):
            # Z < -2.0: Extremely bearish funding = contrarian LONG
            if funding_zscore[i] < -2.0:
                funding_strength = BASE_SIZE
                funding_signal = 1.0  # Long signal
            # Z > +2.0: Extremely bullish funding = contrarian SHORT
            elif funding_zscore[i] > 2.0:
                funding_strength = BASE_SIZE
                funding_signal = -1.0  # Short signal
            # Moderate extremes
            elif funding_zscore[i] < -1.5:
                funding_strength = REDUCED_SIZE
                funding_signal = 1.0
            elif funding_zscore[i] > 1.5:
                funding_strength = REDUCED_SIZE
                funding_signal = -1.0
        
        # === VOLATILITY SPIKE REVERSION (CONFIRMATION) ===
        vol_spike = False
        vol_signal = 0.0
        
        if atr_30[i] > 1e-10:
            atr_ratio = atr_7[i] / atr_30[i]
            
            # Vol spike + price at BB extreme = strong mean reversion signal
            if atr_ratio > 1.8:
                vol_spike = True
                # Long: price below lower BB during vol spike
                if close[i] < bb_lower[i]:
                    vol_signal = 1.0
                # Short: price above upper BB during vol spike
                elif close[i] > bb_upper[i]:
                    vol_signal = -1.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_signal = 0.0
        
        # Long: price breaks Donchian lower (panic low, likely to bounce)
        if close[i] <= donchian_lower[i] * 1.001:  # Small tolerance
            donchian_signal = 1.0
        # Short: price breaks Donchian upper (FOMO high, likely to reject)
        elif close[i] >= donchian_upper[i] * 0.999:
            donchian_signal = -1.0
        
        # === HTF TREND BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === COMBINE SIGNALS ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # Funding is PRIMARY - if extreme, take the trade
        if abs(funding_signal) > 0.5:
            signal_strength = funding_strength
            
            # Add vol spike confirmation
            if vol_spike and np.sign(vol_signal) == funding_signal:
                signal_strength = min(signal_strength * 1.2, MAX_SIZE)
            
            # Apply HTF bias sizing
            if funding_signal > 0:  # Long
                if hma_1d_bull:
                    signal_strength = min(signal_strength * 1.0, MAX_SIZE)
                else:
                    signal_strength = min(signal_strength * 0.7, MAX_SIZE)
            else:  # Short
                if hma_1d_bear:
                    signal_strength = min(signal_strength * 1.0, MAX_SIZE)
                else:
                    signal_strength = min(signal_strength * 0.7, MAX_SIZE)
            
            desired_signal = funding_signal * signal_strength
        
        # If no funding signal, check vol spike + Donchian combo
        elif vol_spike and abs(donchian_signal) > 0.5:
            if np.sign(vol_signal) == donchian_signal:
                signal_strength = REDUCED_SIZE
                
                # Apply HTF bias
                if donchian_signal > 0:  # Long
                    if hma_1d_bull:
                        signal_strength = min(signal_strength * 1.0, MAX_SIZE)
                    else:
                        signal_strength = min(signal_strength * 0.7, MAX_SIZE)
                else:  # Short
                    if hma_1d_bear:
                        signal_strength = min(signal_strength * 1.0, MAX_SIZE)
                    else:
                        signal_strength = min(signal_strength * 0.7, MAX_SIZE)
                
                desired_signal = donchian_signal * signal_strength
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        current_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
        
        if in_position and position_side > 0 and current_atr > 1e-10:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0 and current_atr > 1e-10:
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = current_atr if current_atr > 1e-10 else atr_30[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = current_atr if current_atr > 1e-10 else atr_30[i]
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