#!/usr/bin/env python3
"""
Experiment #1041: 15m Primary + 1h/4h/1d HTF — Fisher Transform + RSI Mean Reversion + HMA Trend

Hypothesis: 15m timeframe needs LOOSER entry conditions to generate trades (learned from 8 failed 15m exps).
Using Fisher Transform (proven reversal indicator in bear markets) + fast RSI(7) + 4h HMA trend filter.
Key insight: RSI thresholds 30-70 work better than 20-80 for trade frequency on lower TF.

Why this should work on 15m:
1. Fisher Transform catches reversals better than RSI alone (Ehlers research)
2. 4h HMA provides trend direction without over-filtering (1d is too slow for 15m entries)
3. RSI(7) is fast enough for 15m but not noise-prone like RSI(3)
4. Session filter 00-12 UTC captures London/NY liquidity (crypto active hours)
5. LOOSE thresholds: Fisher<-1.0/>1.0 + RSI<40/>60 (not extreme 20/80)

CRITICAL FIXES from failed 15m experiments:
- Entry conditions MUST trigger on normal 20-30% moves (not only extremes)
- Use 3 confluence factors max (not 5+ which causes 0 trades)
- Fisher Transform provides reliable reversal signals in bear/range markets
- 4h trend filter (not 1d) matches 15m entry timing better

Position sizing: 0.15-0.20 (smaller for higher TF frequency)
Stoploss: 2.0x ATR(14) trailing
Target: 50-100 trades/year, Sharpe>0.4, DD>-30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_rsi_hma_4h1d_session_v1"
timeframe = "15m"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights reversal points better than RSI in bear/range markets
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (close[i] - lowest) / price_range
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
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
        
        # === HTF TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment (both 4h and 1d agree)
        strong_bull = hma_4h_bull and hma_1d_bull
        strong_bear = hma_4h_bear and hma_1d_bear
        
        # === SESSION FILTER (00-12 UTC preferred, but allow all for trade frequency) ===
        # Loose session filter: prefer 00-12 but allow 12-24 with stronger signals
        is_preferred_session = hours[i] < 12
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Fisher crosses above -1.5 from below = bullish reversal
        fisher_bull_cross = (fisher[i] > -1.5 and fisher_signal[i] <= -1.5) if not np.isnan(fisher_signal[i]) else False
        # Fisher crosses below +1.5 from above = bearish reversal
        fisher_bear_cross = (fisher[i] < 1.5 and fisher_signal[i] >= 1.5) if not np.isnan(fisher_signal[i]) else False
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === ENTRY LOGIC (LOOSE thresholds to guarantee trades) ===
        desired_signal = 0.0
        
        # LONG entries (3 confluence factors max)
        long_score = 0
        if hma_4h_bull:
            long_score += 1
        if rsi_7[i] < 40:
            long_score += 1
        if fisher_oversold or fisher_bull_cross:
            long_score += 1
        if is_preferred_session:
            long_score += 0.5
        
        # SHORT entries
        short_score = 0
        if hma_4h_bear:
            short_score += 1
        if rsi_7[i] > 60:
            short_score += 1
        if fisher_overbought or fisher_bear_cross:
            short_score += 1
        if is_preferred_session:
            short_score += 0.5
        
        # Entry thresholds (LOOSE to ensure trades)
        if long_score >= 2.5:
            if strong_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        elif short_score >= 2.5:
            if strong_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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