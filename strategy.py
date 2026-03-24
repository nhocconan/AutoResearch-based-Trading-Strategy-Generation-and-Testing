#!/usr/bin/env python3
"""
Experiment #961: 15m Primary + 1h/4h/1d HTF — RSI Pullback in Trend

Hypothesis: 15m timeframe with relaxed RSI(7) pullback entries in direction of 4h/1d trend
will generate sufficient trades (50-100/year) while maintaining positive Sharpe.

Key innovations:
1. 4h HMA(21) for intermediate trend bias (not too many HTF filters)
2. 1d HMA(50) for major trend filter (only 2 HTF conditions)
3. 15m RSI(7) for entry timing — LOOSE thresholds (35/65 not 20/80)
4. ATR(14) 2.5x trailing stop for risk management
5. Small position size (0.15-0.25) for 15m frequency
6. Session-agnostic (crypto trades 24/7, no artificial filters)

Why this should work:
- RSI(7) < 35 happens frequently in pullbacks (unlike CRSI < 10)
- Only 2 HTF filters (4h + 1d) not 3+ (1w + 1d + CHOP)
- 15m captures intraday swings with HTF trend alignment
- Relaxed entry thresholds guarantee trades

Entry conditions (LOOSE to guarantee trades):
- LONG = 4h bull + 1d bull + RSI(7) < 40 OR RSI(7) crosses above 30 from below
- SHORT = 4h bear + 1d bear + RSI(7) > 60 OR RSI(7) crosses below 70 from above
- Exit when RSI(7) crosses 50 or stoploss hit

Target: Sharpe>0.3, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
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
        
        # === HTF BIAS (4h + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI CONDITIONS (LOOSE THRESHOLDS FOR TRADES) ===
        rsi_oversold = rsi_7[i] < 40  # Relaxed from 30
        rsi_overbought = rsi_7[i] > 60  # Relaxed from 70
        
        # RSI crossover signals (more frequent entries)
        rsi_cross_up_30 = False
        rsi_cross_down_70 = False
        if i > 0 and not np.isnan(rsi_7[i-1]):
            rsi_cross_up_30 = (rsi_7[i-1] < 30) and (rsi_7[i] >= 30)
            rsi_cross_down_70 = (rsi_7[i-1] > 70) and (rsi_7[i] <= 70)
        
        # RSI crossing 50 (exit signal)
        rsi_cross_above_50 = False
        rsi_cross_below_50 = False
        if i > 0 and not np.isnan(rsi_7[i-1]):
            rsi_cross_above_50 = (rsi_7[i-1] <= 50) and (rsi_7[i] > 50)
            rsi_cross_below_50 = (rsi_7[i-1] >= 50) and (rsi_7[i] < 50)
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG entries - multiple paths to ensure trades
        if htf_4h_bull and htf_1d_bull:
            if rsi_oversold:
                # Deep pullback in uptrend
                desired_signal = SIZE_STRONG
            elif rsi_cross_up_30:
                # RSI recovering from oversold
                desired_signal = SIZE_BASE
            elif rsi_7[i] < 45 and rsi_14[i] < 50:
                # Moderate pullback
                desired_signal = SIZE_BASE
        
        # SHORT entries - multiple paths to ensure trades
        elif htf_4h_bear and htf_1d_bear:
            if rsi_overbought:
                # Deep pullback in downtrend
                desired_signal = -SIZE_STRONG
            elif rsi_cross_down_70:
                # RSI falling from overbought
                desired_signal = -SIZE_BASE
            elif rsi_7[i] > 55 and rsi_14[i] > 50:
                # Moderate pullback
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
            # Exit on RSI cross above 50 (momentum fading)
            if rsi_cross_above_50:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
            # Exit on RSI cross below 50 (momentum fading)
            if rsi_cross_below_50:
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