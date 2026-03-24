#!/usr/bin/env python3
"""
Experiment #573: 5m Primary + 15m/4h HTF — Fisher Transform + Session Filter + Volume Confirmation

Hypothesis: 5m timeframe requires extreme selectivity to avoid fee drag. Using Fisher Transform
(normalizes price to Gaussian, better reversal detection than RSI) for 5m entries, combined with
15m RSI momentum and 4h HMA trend bias. Session filter (08-20 UTC) ensures we only trade during
high liquidity periods (London/NY overlap). Volume confirmation prevents false breakouts.

Key innovations vs failed experiments:
1. Fisher Transform instead of RSI for 5m entries - catches reversals faster
2. Session filter MANDATORY for 5m - only trade 08-20 UTC (high liquidity)
3. Volume ratio confirmation (taker_buy_volume / volume > 0.55 for long)
4. 4h HMA(21) for macro trend - only trade in established direction
5. 15m RSI(14) for medium momentum filter - avoid counter-trend entries
6. Conservative sizing (0.15-0.25) due to higher trade frequency on 5m
7. ATR(14)*2.5 stoploss on all positions

Strategy logic:
1. 4h HMA(21) = macro trend bias (align_htf_to_ltf for proper alignment)
2. 15m RSI(14) = medium momentum (RSI>50 for long bias, RSI<50 for short bias)
3. 5m Fisher(9) = entry timing (Fisher crosses -1.5 for long, +1.5 for short)
4. Session filter = only 08:00-20:00 UTC (London/NY high liquidity)
5. Volume ratio = taker_buy_volume/volume > 0.55 for long, < 0.45 for short
6. ATR(14)*2.5 stoploss - signal→0 when stop hit

Target: Sharpe>0.40, trades>=50/year (200+ train, 50+ test), DD<-30%
Timeframe: 5m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_fisher_session_volume_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at catching turning points than RSI, works in both trending/ranging markets
    
    Formula:
    1. Calculate typical price: (high + low + close) / 3
    2. Normalize: (price - lowest_low) / (highest_high - lowest_low)
    3. Transform: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    # Normalize price over lookback period
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized[i] = 0.667 * (typical[i] - lowest) / price_range - 0.333
            # Clamp to avoid division by zero in log
            normalized[i] = max(-0.999, min(0.999, normalized[i]))
        else:
            normalized[i] = 0.0
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        if abs(normalized[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
        else:
            fisher[i] = fisher[i-1] if i > period else 0.0
    
    # Smooth with EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher_smooth

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for macro trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for medium momentum
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    rsi_5m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 1e-10:
            volume_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(rsi_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        # Convert open_time (milliseconds) to hour
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4H MACRO TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15M MOMENTUM FILTER ===
        momentum_bull = rsi_15m_aligned[i] > 50.0
        momentum_bear = rsi_15m_aligned[i] < 50.0
        
        # === 5M FISHER ENTRY SIGNALS ===
        # Fisher crosses above -1.5 from below = long entry
        # Fisher crosses below +1.5 from above = short entry
        fisher_long_signal = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_short_signal = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # Fisher extreme oversold/overbought
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === VOLUME CONFIRMATION ===
        volume_bull = volume_ratio[i] > 0.55  # More buying pressure
        volume_bear = volume_ratio[i] < 0.45  # More selling pressure
        
        # === RSI 5M FILTER ===
        rsi_5m_bull = rsi_5m[i] > 40.0 and rsi_5m[i] < 70.0
        rsi_5m_bear = rsi_5m[i] > 30.0 and rsi_5m[i] < 60.0
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bull + 15m momentum + Fisher signal + volume
        if htf_bull and momentum_bull:
            confluence_count = 0
            if fisher_long_signal:
                confluence_count += 2
            if fisher_extreme_long:
                confluence_count += 1
            if volume_bull:
                confluence_count += 1
            if rsi_5m_bull:
                confluence_count += 1
            
            if confluence_count >= 3:
                desired_signal = SIZE_STRONG
            elif confluence_count >= 2:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 4h bear + 15m momentum + Fisher signal + volume
        elif htf_bear and momentum_bear:
            confluence_count = 0
            if fisher_short_signal:
                confluence_count += 2
            if fisher_extreme_short:
                confluence_count += 1
            if volume_bear:
                confluence_count += 1
            if rsi_5m_bear:
                confluence_count += 1
            
            if confluence_count >= 3:
                desired_signal = -SIZE_STRONG
            elif confluence_count >= 2:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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