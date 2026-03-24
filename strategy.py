#!/usr/bin/env python3
"""
Experiment #026: 12h Primary + 1d HTF — Relaxed RSI Dual Regime with Funding

Hypothesis: After 22 failed experiments, the key lesson is ENTRY CONDITIONS MUST BE LOOSE ENOUGH.
Strategies #022, #023, #025 all got Sharpe=0.000 (zero trades) due to overly strict filters.

Key insight from #019 (Sharpe=0.368): 4h RSI(14) + Choppiness + Funding + 1d HTF worked.
This adapts that proven formula to 12h timeframe with RELAXED thresholds:

1. RSI(14) with 30/70 thresholds (NOT extreme 15/85) - generates more signals
2. Choppiness > 45 = choppy (mean revert), < 40 = trending (momentum)
3. 1d HMA(21) for trend bias - simpler than dual 1d/1w
4. Funding rate contrarian overlay - proven BTC/ETH edge
5. Loose regime detection - neutral zone 40-45 allows both strategies

Why this should work on 12h:
- RSI 30/70 triggers ~25% of bars (vs ~10% for 15/85)
- Choppiness 45 threshold (vs 55) = more bars qualify as choppy
- 12h has 730 bars/year, need 4-8% signal rate = 30-60 trades/year
- Discrete sizing (0.20, 0.30) minimizes fee churn

Risk Management:
- 2.5x ATR trailing stop
- Max position 0.35 (35% of capital)
- Discrete levels: 0.0, ±0.20, ±0.30

Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_chop_funding_1d_relaxed_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Standard RSI with proper min_periods"""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection (lower = more trending)"""
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

def calculate_hma(close, period=21):
    """Hull Moving Average for HTF trend"""
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

def load_funding_data(prices):
    """Load funding rate data - tries symbol-specific path"""
    try:
        import os
        # Try to get symbol from prices DataFrame
        symbol = None
        if hasattr(prices, 'attrs') and 'symbol' in prices.attrs:
            symbol = prices.attrs['symbol']
        
        if symbol is None:
            symbol = "BTCUSDT"  # Default fallback
        
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
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Load funding data
    funding_data = load_funding_data(prices)
    
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
    
    for i in range(350, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # RELAXED thresholds for more trade generation
        is_choppy = chop[i] > 45.0  # Lower threshold = more bars qualify
        is_trending = chop[i] < 40.0  # Clear trending only below 40
        
        # === HTF TREND BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FUNDING RATE CONTRARIAN ===
        funding_adjustment = 0.0
        try:
            funding_rate = get_funding_at_time(funding_data, open_time[i])
            if funding_rate < -0.005:  # Very negative funding = contrarian long
                funding_adjustment = 0.05
            elif funding_rate > 0.005:  # Very positive funding = contrarian short
                funding_adjustment = -0.05
        except Exception:
            funding_adjustment = 0.0
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - RSI extremes (RELAXED 30/70 thresholds)
            # Long: RSI < 30 (oversold in choppy market)
            if rsi[i] < 30.0:
                if hma_1d_bull:
                    desired_signal = BASE_SIZE + funding_adjustment
                else:
                    desired_signal = REDUCED_SIZE + funding_adjustment
            
            # Short: RSI > 70 (overbought in choppy market)
            elif rsi[i] > 70.0:
                if hma_1d_bear:
                    desired_signal = -BASE_SIZE + funding_adjustment
                else:
                    desired_signal = -REDUCED_SIZE + funding_adjustment
        
        elif is_trending:
            # TREND REGIME - RSI momentum with HTF confirmation
            # Long: RSI > 50 (momentum up) + HTF bullish
            if rsi[i] > 50.0:
                if hma_1d_bull:
                    desired_signal = BASE_SIZE + funding_adjustment
                else:
                    desired_signal = REDUCED_SIZE + funding_adjustment
            
            # Short: RSI < 50 (momentum down) + HTF bearish
            elif rsi[i] < 50.0:
                if hma_1d_bear:
                    desired_signal = -BASE_SIZE + funding_adjustment
                else:
                    desired_signal = -REDUCED_SIZE + funding_adjustment
        
        else:
            # NEUTRAL REGIME (40 <= CHOP <= 45) - trade either direction with HTF bias
            if hma_1d_bull and rsi[i] > 45.0:
                desired_signal = REDUCED_SIZE + funding_adjustment
            elif hma_1d_bear and rsi[i] < 55.0:
                desired_signal = -REDUCED_SIZE + funding_adjustment
        
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