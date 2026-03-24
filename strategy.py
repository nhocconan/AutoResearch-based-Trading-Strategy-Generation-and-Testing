#!/usr/bin/env python3
"""
Experiment #024: 4h Primary + 12h/1d HTF — Dual Regime Breakout with RSI Pullback

Hypothesis: Building on #019's success (Sharpe=0.368), this simplifies entry logic
to ensure trade generation while maintaining edge. Key changes:

1. DONCHIAN BREAKOUT + HMA TREND: Proven pattern for SOL (Sharpe +0.782 in research)
   - Long: Price breaks Donchian(20) high + price > HMA(21)
   - Short: Price breaks Donchian(20) low + price < HMA(21)
   
2. CHOPPINESS REGIME FILTER: Only trade breakouts when CHOP < 50 (trending regime)
   - When CHOP > 60: Switch to RSI mean reversion (oversold/overbought)
   
3. RSI PULLBACK ENTRY: Instead of breakout at extreme, wait for RSI(7) pullback to 40-60
   - Reduces false breakouts, improves win rate
   
4. 12h HMA TREND BIAS: Stronger filter than 1d for 4h timeframe
   - Only long when 12h HMA sloping up, only short when sloping down
   
5. FUNDING CONTRARIAN: Keep proven BTC/ETH edge from #019
   - Adds 0.10 signal when funding extreme

Why this should beat #019:
- Simpler entry logic = more trades (avoid 0-trade failure)
- Donchian breakout = catches major moves (2021 bull, 2022 crash)
- Choppiness filter = avoids whipsaw in range markets
- RSI pullback = better entry timing than pure breakout
- Expected: 35-50 trades/year, Sharpe > 0.4

Entry Logic:
- TRENDING (CHOP<50): Donchian breakout + HMA confirmation + RSI pullback
- CHOPPY (CHOP>60): RSI(7) < 25 long, RSI(7) > 75 short
- Size: 0.30 with 12h trend alignment, 0.20 neutral
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_rsi_pullback_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_rsi(close, period=7):
    """RSI with configurable period"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection (0-100, high=choppy)"""
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for major trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 4h HMA for local trend
    hma_4h = calculate_hma(close, period=21)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trending = chop[i] < 50.0
        is_choppy = chop[i] > 60.0
        
        # === HTF TREND BIAS (12h and 1d) ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 12h HMA slope (5-bar lookback)
        hma_12h_slope = 0.0
        if i >= 5 and not np.isnan(hma_12h_aligned[i-5]):
            hma_12h_slope = (hma_12h_aligned[i] - hma_12h_aligned[i-5]) / hma_12h_aligned[i-5] if hma_12h_aligned[i-5] > 1e-10 else 0.0
        
        # 4h HMA slope for local momentum
        hma_4h_slope = 0.0
        if i >= 5 and not np.isnan(hma_4h[i-5]):
            hma_4h_slope = (hma_4h[i] - hma_4h[i-5]) / hma_4h[i-5] if hma_4h[i-5] > 1e-10 else 0.0
        
        # === FUNDING RATE CONTRARIAN ===
        funding_signal = 0.0
        try:
            funding_rate = get_funding_at_time(funding_data, open_time[i])
            if funding_rate > 0.01:
                funding_signal = -0.10
            elif funding_rate < -0.01:
                funding_signal = 0.10
        except Exception:
            funding_signal = 0.0
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # Count HTF alignment
        htf_bull_count = sum([hma_12h_bull, hma_1d_bull])
        htf_bear_count = sum([hma_12h_bear, hma_1d_bear])
        
        if is_trending:
            # TREND REGIME: Donchian breakout + HMA confirmation + RSI pullback
            
            # LONG: Breakout above Donchian + price > HMA + RSI pullback to 40-60
            if close[i] > donchian_upper[i] and close[i] > hma_4h[i]:
                if 35 <= rsi[i] <= 65:  # RSI pullback zone (not overbought)
                    if hma_12h_slope > 0 or htf_bull_count >= 2:
                        signal_strength = BASE_SIZE
                    elif htf_bull_count >= 1:
                        signal_strength = REDUCED_SIZE
                    desired_signal = signal_strength + funding_signal
            
            # SHORT: Breakout below Donchian + price < HMA + RSI pullback to 35-65
            elif close[i] < donchian_lower[i] and close[i] < hma_4h[i]:
                if 35 <= rsi[i] <= 65:  # RSI pullback zone (not oversold)
                    if hma_12h_slope < 0 or htf_bear_count >= 2:
                        signal_strength = BASE_SIZE
                    elif htf_bear_count >= 1:
                        signal_strength = REDUCED_SIZE
                    desired_signal = -signal_strength + funding_signal
        
        elif is_choppy:
            # CHOPPY REGIME: RSI mean reversion (looser thresholds for trade gen)
            
            # LONG: RSI < 30 (oversold in range)
            if rsi[i] < 30.0:
                if htf_bull_count >= 1:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength + funding_signal
            
            # SHORT: RSI > 70 (overbought in range)
            elif rsi[i] > 70.0:
                if htf_bear_count >= 1:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength + funding_signal
        
        else:
            # NEUTRAL REGIME (50 <= CHOP <= 60): Only trade with strong HTF trend
            if htf_bull_count >= 2 and hma_4h_slope > 0.001 and rsi[i] < 60:
                desired_signal = REDUCED_SIZE + funding_signal
            elif htf_bear_count >= 2 and hma_4h_slope < -0.001 and rsi[i] > 40:
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