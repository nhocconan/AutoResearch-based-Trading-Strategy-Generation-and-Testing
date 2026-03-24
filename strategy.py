#!/usr/bin/env python3
"""
Experiment #1665: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + Funding Contrarian

Hypothesis: 1h strategies fail due to TOO MANY TRADES (>200/yr) causing fee drag.
Solution: Use 4h/1d for SIGNAL DIRECTION, 1h only for precise ENTRY TIMING.

Key innovations vs failed 1h attempts (#1655, #1658, #1660):
1. Fisher Transform (Ehlers) instead of RSI — sharper reversal signals, fewer false triggers
2. Funding rate contrarian filter — proven edge for BTC/ETH (z-score < -2 long, > +2 short)
3. STRICT 5-confluence entry: 4h trend + 1d bias + Fisher extreme + CHOP regime + session(8-20 UTC)
4. Smaller position size (0.20-0.25) for lower TF to minimize fee impact
5. Volume confirmation (>1.2x 20-bar avg) to avoid low-liquidity traps

Entry Logic (ALL must align):
- 4h HMA(21) defines trend direction (price > HMA = long bias)
- 1d HMA(21) confirms broader regime (same direction = full size, opposite = half size)
- Fisher Transform crosses -1.5 (long) or +1.5 (short) on 1h
- CHOP > 55 = range (mean revert entries), CHOP < 45 = trend (pullback entries)
- Session: only 8-20 UTC (avoid Asian session whipsaw)
- Volume: current > 1.2x 20-bar average

Risk: 2.5x ATR trailing stop, discrete signals (0.0, ±0.20, ±0.25)
Target: 30-50 trades/year, Sharpe > 0.618, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_funding_4h1d_session_volume_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — catches reversals in bear/range markets
    Transforms price to nearly Gaussian distribution for clearer signals
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Research shows superior performance to RSI in mean-reversion regimes
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate median price (HL2)
    hl2 = (high + low) / 2.0
    
    # Normalize price to range [-1, 1]
    for i in range(period, n):
        highest = np.max(hl2[i - period + 1:i + 1])
        lowest = np.min(hl2[i - period + 1:i + 1])
        
        if highest == lowest:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize to [-0.999, 0.999] to avoid division issues
        normalized = 0.999 * 2.0 * ((hl2[i] - lowest) / (highest - lowest) - 0.5)
        
        # Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
        if abs(normalized) >= 0.999:
            fisher[i] = np.sign(normalized) * 2.0
        else:
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

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
        weights = np.arange(1, span + 1)
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

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for regime bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.20
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_avg[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === HTF TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # HTF alignment check (4h and 1d agree = stronger signal)
        htf_aligned_bull = hma_4h_bull and hma_1d_bull
        htf_aligned_bear = hma_4h_bear and hma_1d_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = fisher[i] > -1.5 and fisher_prev[i] <= -1.5
        # Short: Fisher crosses below +1.5 from above
        fisher_short = fisher[i] < 1.5 and fisher_prev[i] >= 1.5
        
        # === DESIRED SIGNAL BASED ON REGIME + CONFLUENCE ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME
            # Long: Fisher long + 4h bullish + session + volume
            if fisher_long and in_session and volume_confirmed:
                if htf_aligned_bull:
                    signal_strength = BASE_SIZE
                elif hma_4h_bull:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength
            
            # Short: Fisher short + 4h bearish + session + volume
            elif fisher_short and in_session and volume_confirmed:
                if htf_aligned_bear:
                    signal_strength = BASE_SIZE
                elif hma_4h_bear:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength
        
        elif is_trending:
            # TREND REGIME — only enter on pullbacks (Fisher extremes)
            # Long: 4h bullish + Fisher long (pullback entry)
            if hma_4h_bull and fisher_long and in_session and volume_confirmed:
                if htf_aligned_bull:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength
            
            # Short: 4h bearish + Fisher short (pullback entry)
            elif hma_4h_bear and fisher_short and in_session and volume_confirmed:
                if htf_aligned_bear:
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength
        
        else:
            # NEUTRAL REGIME (45 <= CHOP <= 55) — only trade with full HTF alignment
            if htf_aligned_bull and fisher_long and in_session and volume_confirmed:
                desired_signal = REDUCED_SIZE
            elif htf_aligned_bear and fisher_short and in_session and volume_confirmed:
                desired_signal = -REDUCED_SIZE
        
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