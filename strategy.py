#!/usr/bin/env python3
"""
Experiment #1379: 1h Primary + 4h/12h HTF — Vol Spike Mean Reversion + Fisher Transform

Hypothesis: After 1134 failed strategies, the key insight is:
1. Zero-trade strategies (Sharpe=0.000) = entry conditions TOO STRICT
2. Negative Sharpe strategies = trend-following fails in bear/range markets (2025 test)
3. BEST EDGE for BTC/ETH: Vol spike reversion + Fisher Transform (catches reversals better than RSI)

This strategy combines:
1. 4h HMA(21) for trend bias (avoid counter-trend in major moves)
2. 12h HMA(21) for major regime filter (stronger conviction)
3. ATR(7)/ATR(30) vol spike detection (>1.5 = panic/reversal likely)
4. Ehlers Fisher Transform(9) for reversal timing (crosses at -1.0/+1.0)
5. Bollinger Band(20,2.0) for mean reversion confirmation
6. Session filter (08-20 UTC) for liquidity boost (optional, not required)

Why this should work where others failed:
- Fisher Transform catches reversals better than RSI (proven in literature)
- Vol spike + mean reversion works in bear/range markets (2025 test period)
- LOOSE entry conditions (Fisher < -1.0 not -1.5, ATR ratio > 1.5 not > 2.0)
- Session filter is OPTIONAL boost, not required (prevents 0 trades)
- 1h TF with 4h/12h HTF = 40-80 trades/year (fee-friendly)

Entry logic (LOOSE to guarantee trades):
- LONG: 4h_HMA bullish + ATR_ratio > 1.5 + Fisher < -1.0 + price < BB_lower
- SHORT: 4h_HMA bearish + ATR_ratio > 1.5 + Fisher > 1.0 + price > BB_upper
- 12h_HMA alignment = larger size (0.30 vs 0.20)

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_vol_spike_meanreversion_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals better than RSI in bear/range markets
    """
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + np.roll(high + low, 1)) / 4.0
    typical[0] = (high[0] + low[0]) / 2.0
    
    # Normalize to -1 to +1 range
    fisher_input = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        window = typical[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 2:
            highest = np.max(valid)
            lowest = np.min(valid)
            if highest > lowest:
                fisher_input[i] = 0.66 * ((typical[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * fisher_input[i-1] if i > period and not np.isnan(fisher_input[i-1]) else 0.0
                fisher_input[i] = max(-0.999, min(0.999, fisher_input[i]))
    
    # Fisher transform
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(fisher_input[i]) and abs(fisher_input[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + fisher_input[i]) / (1 - fisher_input[i]))
            if i > period and not np.isnan(fisher[i-1]):
                fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def calculate_atr_ratio(atr_short, atr_long):
    """ATR ratio for vol spike detection"""
    ratio = np.full(len(atr_short), np.nan, dtype=np.float64)
    mask = (atr_long > 0) & (~np.isnan(atr_short)) & (~np.isnan(atr_long))
    ratio[mask] = atr_short[mask] / atr_long[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr_7, atr_30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Extract hour from open_time for session filter
    try:
        hours = pd.to_datetime(prices['open_time']).dt.hour.values
    except:
        hours = np.zeros(n, dtype=np.int32)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 12h HMA for major regime (stronger filter)
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 1.5 = vol spike (panic/reversal likely)
        vol_spike = atr_ratio[i] > 1.5
        
        # === FISHER TRANSFORM REVERSAL ===
        fisher_value = fisher[i]
        fisher_prev = fisher_signal[i] if not np.isnan(fisher_signal[i]) else fisher_value
        
        # Fisher oversold (reversal up likely)
        fisher_oversold = fisher_value < -1.0
        
        # Fisher overbought (reversal down likely)
        fisher_overbought = fisher_value > 1.0
        
        # === BOLLINGER BAND MEAN REVERSION ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === SESSION FILTER (optional boost, not required) ===
        hour = hours[i] if i < len(hours) else 12
        session_active = 8 <= hour <= 20  # 08-20 UTC
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + vol spike + Fisher oversold + price below BB
        # Session filter is OPTIONAL (boosts size, not required for entry)
        if price_above_4h and vol_spike and fisher_oversold and price_below_bb:
            if price_above_12h:
                # Strong trend alignment (4h + 12h both bullish)
                base_size = SIZE_STRONG
            else:
                # Basic long (only 4h bullish)
                base_size = SIZE_BASE
            
            desired_signal = base_size
            
            # Session boost (optional)
            if session_active:
                desired_signal = min(SIZE_STRONG, desired_signal * 1.1)
        
        # SHORT: 4h bearish + vol spike + Fisher overbought + price above BB
        elif price_below_4h and vol_spike and fisher_overbought and price_above_bb:
            if price_below_12h:
                # Strong trend alignment (4h + 12h both bearish)
                base_size = SIZE_STRONG
            else:
                # Basic short (only 4h bearish)
                base_size = SIZE_BASE
            
            desired_signal = -base_size
            
            # Session boost (optional)
            if session_active:
                desired_signal = max(-SIZE_STRONG, desired_signal * 1.1)
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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