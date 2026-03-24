#!/usr/bin/env python3
"""
Experiment #669: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + ADX Filter

Hypothesis: 15m timeframe needs VERY selective entries to avoid fee drag (>100 trades/yr kills PnL).
Using 1d HMA for primary bias + 1h ADX for trend strength + 15m RSI pullback for entry timing.
This is the 15m adaptation of the successful 6h regime strategy but with stricter filters.

Key innovations:
1. 1d HMA(21) bias - only long above, only short below (strong HTF filter)
2. 1h ADX(14) > 18 - only trade when trending (avoid 15m chop)
3. 15m RSI(14) pullback - enter on retracement (35-45 long, 55-65 short), not breakout
4. Position size 0.15-0.20 - smaller for higher frequency (15m)
5. ATR(14) 2.5x trailing stop - tight risk management
6. Target 50-80 trades/year - strict enough to avoid fee drag

Why this might work on 15m:
- HTF filters (1d/1h) reduce trade frequency to HTF levels
- 15m only used for entry timing precision
- RSI pullback more selective than HMA cross
- Smaller size compensates for higher frequency

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-30%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_adx_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response with less lag"""
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
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    adx_1h_raw = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, period=14)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_15m = calculate_rsi(close, period=14)
    atr_15m = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === HTF TREND STRENGTH (1h ADX) ===
        trend_strong = adx_1h_aligned[i] > 18.0
        
        # === 15m HMA DIRECTION ===
        hma_15m_bull = False
        hma_15m_bear = False
        if i >= 3 and not np.isnan(hma_15m[i-3]):
            hma_15m_bull = hma_15m[i] > hma_15m[i-1] and hma_15m[i-1] > hma_15m[i-2]
            hma_15m_bear = hma_15m[i] < hma_15m[i-1] and hma_15m[i-1] < hma_15m[i-2]
        
        # === 15m RSI PULLBACK (key entry trigger) ===
        # Long: RSI pulled back to 35-50 zone (oversold but not extreme)
        # Short: RSI pulled back to 50-65 zone (overbought but not extreme)
        rsi_pullback_long = 35.0 <= rsi_15m[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi_15m[i] <= 65.0
        
        # === PRICE vs HMA CONFIRMATION ===
        price_above_hma = close[i] > hma_15m[i]
        price_below_hma = close[i] < hma_15m[i]
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 1h ADX strong + 15m RSI pullback + price above 15m HMA
        if htf_bull and trend_strong and rsi_pullback_long and price_above_hma:
            desired_signal = SIZE_STRONG
        elif htf_bull and trend_strong and rsi_pullback_long:
            desired_signal = SIZE_BASE
        elif htf_bull and hma_15m_bull and price_above_hma:
            # Weaker: just alignment without RSI pullback
            desired_signal = SIZE_BASE * 0.5
        
        # SHORT: 1d bear + 1h ADX strong + 15m RSI pullback + price below 15m HMA
        elif htf_bear and trend_strong and rsi_pullback_short and price_below_hma:
            desired_signal = -SIZE_STRONG
        elif htf_bear and trend_strong and rsi_pullback_short:
            desired_signal = -SIZE_BASE
        elif htf_bear and hma_15m_bear and price_below_hma:
            # Weaker: just alignment without RSI pullback
            desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_15m[i]
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