#!/usr/bin/env python3
"""
Experiment #585: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 4h HMA trend filter + 1d macro bias + RSI(7) pullback
entries during London/NY session (00-12 UTC) will capture intraday momentum while
avoiding Asian session noise. This is the FIRST 15m experiment - critical to get
trade frequency right (target 40-100 trades/year, not 0 trades like #577/#581).

Key design decisions:
1. 4h HMA(21) = trend direction filter (HTF bias)
2. 1d HMA(21) = macro bias (only trade with daily trend)
3. 15m RSI(7) = entry timing (faster than RSI(14), catches pullbacks)
4. Session filter = 00-12 UTC only (London+NY overlap, avoids Asian chop)
5. Position size = 0.20 (smaller for 15m frequency, reduces fee drag)
6. ATR(14)*2.5 stoploss on all positions
7. Entry conditions LOOSE enough to generate trades (RSI 30-45 long, 55-70 short)

Why this might work on 15m:
- HTF filters prevent trading against major trend
- Session filter cuts 50% of noise (Asian session = whipsaw)
- RSI(7) is sensitive enough to catch intraday pullbacks
- Small position size (0.20) survives 2022-style crashes

CRITICAL: Must generate >10 trades on train, >3 on test. Previous 15m attempts
failed with Sharpe=0.000 (zero trades). Entry thresholds are intentionally loose.

Target: Sharpe>0.40, trades>=40 train (10/year), trades>=5 test
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_4h1d_v1"
timeframe = "15m"
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    # Convert to hours UTC
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    
    # Also calculate 15m HMA for local trend
    hma_15m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.20  # Smaller size for 15m frequency
    
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
        
        if np.isnan(rsi[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Get UTC hour for session filter
        hour = get_session_hour(open_time[i])
        
        # Session filter: 00-12 UTC (London + NY overlap)
        # This avoids Asian session chop (12-00 UTC)
        in_good_session = (hour >= 0 and hour < 12)
        
        # === HTF BIAS (1d macro + 4h trend) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        
        # === 15m LOCAL TREND ===
        local_bull = close[i] > hma_15m[i]
        local_bear = close[i] < hma_15m[i]
        
        # === RSI ZONES (loose thresholds to ensure trades) ===
        # Long: RSI 30-45 (oversold but not extreme)
        # Short: RSI 55-70 (overbought but not extreme)
        rsi_long_zone = rsi[i] >= 30.0 and rsi[i] <= 45.0
        rsi_short_zone = rsi[i] >= 55.0 and rsi[i] <= 70.0
        
        # RSI turning up/down
        rsi_turning_up = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_turning_down = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI in long zone + turning up + good session
        if htf_bull and rsi_long_zone and rsi_turning_up:
            if in_good_session:
                desired_signal = SIZE
            else:
                # Allow entries outside session but smaller size
                desired_signal = SIZE * 0.6
        
        # SHORT: HTF bear + RSI in short zone + turning down + good session
        elif htf_bear and rsi_short_zone and rsi_turning_down:
            if in_good_session:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE * 0.6
        
        # Additional entry: RSI extreme reversal (catches big moves)
        if rsi[i] < 25.0 and rsi_turning_up and htf_bull:
            desired_signal = SIZE
        
        if rsi[i] > 75.0 and rsi_turning_down and htf_bear:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trailing stop
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trailing stop
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif abs(desired_signal) >= SIZE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE
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