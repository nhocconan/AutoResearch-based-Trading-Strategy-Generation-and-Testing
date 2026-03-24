#!/usr/bin/env python3
"""
Experiment #507: 6h Primary + 1d HTF — Simplified Trend + Funding Contrarian

Hypothesis: 6h timeframe needs cleaner signals with fewer overlapping conditions.
Current #495 has too many entry triggers causing signal dilution (Sharpe=0.026).
New approach: 1d HMA trend bias + 6h RSI(7) fast mean reversion + funding z-score filter.

Key changes from #495:
1. RSI(7) instead of RSI(14) — faster response for 6h bars
2. Funding rate z-score as contrarian filter (BEST EDGE for BTC/ETH per research)
3. Asymmetric sizing: 0.30 with HTF trend, 0.20 counter-trend
4. Fewer entry conditions (3 core triggers, not 8+)
5. Clearer hysteresis to reduce signal churn
6. 2.5x ATR trailing stoploss on all positions

Strategy logic:
1. 1d HMA(21) = daily trend bias (HTF filter)
2. 6h RSI(7) extremes = entry triggers (25/75 thresholds for 6h speed)
3. Funding z-score(30) = contrarian filter (short when >+2, long when <-2)
4. ATR(14) = dynamic stoploss (2.5x from entry, trailed)
5. Size: 0.30 with trend, 0.20 counter-trend, 0.0 flat

Target: Sharpe>0.40, trades>=120 train, trades>=20 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_rsi7_funding_hma_1d_v1"
timeframe = "6h"
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

def calculate_zscore(series, period=30):
    """Z-score of a series"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    rolling_mean = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(series).rolling(window=period, min_periods=period).std().values
    
    zscore = (series - rolling_mean) / (rolling_std + 1e-10)
    return zscore

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def load_funding_data(symbol):
    """Load funding rate data for contrarian filter"""
    try:
        # Map symbol to funding file path
        symbol_map = {
            'BTCUSDT': 'data/processed/funding/BTCUSDT.parquet',
            'ETHUSDT': 'data/processed/funding/ETHUSDT.parquet',
            'SOLUSDT': 'data/processed/funding/SOLUSDT.parquet'
        }
        if symbol in symbol_map:
            df = pd.read_parquet(symbol_map[symbol])
            return df['funding_rate'].values
    except:
        pass
    return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol from prices metadata if available
    symbol = prices.get('symbol', 'BTCUSDT') if hasattr(prices, 'get') else 'BTCUSDT'
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Load funding rate data for contrarian filter
    funding_rates = load_funding_data(symbol)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_zscore = calculate_zscore(funding_rates[:n], period=30)
    else:
        funding_zscore = np.full(n, np.nan)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_fast = calculate_rsi(close, period=7)  # Faster RSI for 6h
    rsi_slow = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_WITH_TREND = 0.30
    SIZE_COUNTER_TREND = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi_fast[i]) or np.isnan(rsi_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d HTF BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI EXTREMES (FAST RSI(7): 25/75 thresholds) ===
        rsi_extreme_oversold = rsi_fast[i] < 25.0
        rsi_extreme_overbought = rsi_fast[i] > 75.0
        rsi_oversold = rsi_fast[i] < 35.0
        rsi_overbought = rsi_fast[i] > 65.0
        rsi_rising = rsi_fast[i] > rsi_fast[i-1] if i > 0 else False
        rsi_falling = rsi_fast[i] < rsi_fast[i-1] if i > 0 else False
        
        # === FUNDING Z-SCORE CONTRARIAN ===
        funding_extreme_long = not np.isnan(funding_zscore[i]) and funding_zscore[i] < -2.0
        funding_extreme_short = not np.isnan(funding_zscore[i]) and funding_zscore[i] > 2.0
        
        # === ENTRY LOGIC (CLEANER - 3 core triggers) ===
        desired_signal = 0.0
        
        # TRIGGER 1: HTF trend + RSI pullback entry (primary signal)
        if htf_bull and hma_bull and above_sma50:
            if rsi_extreme_oversold or (rsi_oversold and rsi_rising):
                desired_signal = SIZE_WITH_TREND
        elif htf_bear and hma_bear and below_sma50:
            if rsi_extreme_overbought or (rsi_overbought and rsi_falling):
                desired_signal = -SIZE_WITH_TREND
        
        # TRIGGER 2: Funding contrarian (works against HTF trend too)
        if desired_signal == 0.0:
            if funding_extreme_long and rsi_oversold:
                desired_signal = SIZE_COUNTER_TREND
            elif funding_extreme_short and rsi_overbought:
                desired_signal = -SIZE_COUNTER_TREND
        
        # TRIGGER 3: SMA200 mean reversion (strong reversal signal)
        if desired_signal == 0.0:
            if rsi_extreme_oversold and above_sma200 and rsi_rising:
                desired_signal = SIZE_COUNTER_TREND
            elif rsi_extreme_overbought and below_sma200 and rsi_falling:
                desired_signal = -SIZE_COUNTER_TREND
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, high[i])
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: move up as price rises
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: move down as price falls
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_WITH_TREND * 0.9:
            final_signal = SIZE_WITH_TREND
        elif desired_signal <= -SIZE_WITH_TREND * 0.9:
            final_signal = -SIZE_WITH_TREND
        elif desired_signal >= SIZE_COUNTER_TREND * 0.9:
            final_signal = SIZE_COUNTER_TREND
        elif desired_signal <= -SIZE_COUNTER_TREND * 0.9:
            final_signal = -SIZE_COUNTER_TREND
        else:
            final_signal = 0.0
        
        # === HYSTERESIS: Don't flip without flat period ===
        if in_position and final_signal != 0.0 and np.sign(final_signal) != position_side:
            # Going from long to short or vice versa - go flat first
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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