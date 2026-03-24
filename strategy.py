#!/usr/bin/env python3
"""
Experiment #192: 12h Primary + 1d HTF — Simplified RSI Mean Reversion with HTF Trend

Hypothesis: Previous 12h strategies failed due to overly complex regime detection
with too many conflicting filters (Choppiness + Connors RSI + Donchian = 0 trades).
This version simplifies to core principles that generate trades:

1. 1d HMA(50) = Major trend direction (only trade WITH HTF trend)
2. 12h RSI(14) = Entry timing (oversold in uptrend, overbought in downtrend)
3. ATR(14) = Stoploss (2.5x trailing)
4. Funding rate z-score = Additional filter for BTC/ETH mean reversion

Key changes from #172:
- REMOVED: Choppiness Index (transition zones killed trades)
- REMOVED: Connors RSI (too rare, complex calculation)
- REMOVED: Donchian breakout (conflicts with mean reversion)
- SIMPLIFIED: RSI(14) < 40 long, > 60 short (more frequent signals)
- ADDED: Funding rate filter for BTC/ETH (proven edge from research notes)

Expected: 40-60 trades/year on 12h, Sharpe > 0.4, DD < -35%
Position sizing: 0.30 base, 0.25 reduced (discrete levels to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_htf_trend_funding_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    mask = avg_loss > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + avg_gain[mask] / avg_loss[mask]))
    rsi[~mask] = 100.0
    
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

def calculate_zscore(series, period=30):
    """Rolling Z-score for mean reversion signals"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    rolling_mean = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(series).rolling(window=period, min_periods=period).std().values
    
    zscore = (series - rolling_mean) / (rolling_std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators - vectorized before loop
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_12h = calculate_hma(close, period=21)
    
    # Try to load funding rate data (BTC/ETH edge from research)
    funding_zscore = None
    try:
        # Funding data path pattern from research notes
        symbol = prices.get('symbol', 'BTCUSDT')
        funding_path = f"data/processed/funding/{symbol}.parquet"
        funding_df = pd.read_parquet(funding_path)
        if len(funding_df) >= n:
            funding_rates = funding_df['funding_rate'].values[:n]
            funding_zscore = calculate_zscore(funding_rates, period=30)
    except:
        funding_zscore = None  # SOL or missing data - trade without funding filter
    
    signals = np.zeros(n)
    SIZE_BASE = 0.30  # 30% base position size
    SIZE_REDUCED = 0.25  # 25% for weaker signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI EXTREMES (looser thresholds for more trades) ===
        rsi_oversold = rsi[i] < 40.0  # Was 35, loosened for more trades
        rsi_overbought = rsi[i] > 60.0  # Was 65, loosened for more trades
        
        # === FUNDING RATE FILTER (BTC/ETH only) ===
        funding_long_ok = True
        funding_short_ok = True
        if funding_zscore is not None and not np.isnan(funding_zscore[i]):
            funding_long_ok = funding_zscore[i] < 1.0  # Not extremely long-biased
            funding_short_ok = funding_zscore[i] > -1.0  # Not extremely short-biased
        
        # === ENTRY LOGIC (simplified for trade generation) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI oversold + funding not extreme long
        if htf_bull and rsi_oversold and funding_long_ok:
            desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + RSI overbought + funding not extreme short
        elif htf_bear and rsi_overbought and funding_short_ok:
            desired_signal = -SIZE_BASE
        
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
        if desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_REDUCED * 0.9:
            final_signal = SIZE_REDUCED
        elif desired_signal <= -SIZE_REDUCED * 0.9:
            final_signal = -SIZE_REDUCED
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
                # Flip position
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