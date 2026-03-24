#!/usr/bin/env python3
"""
Experiment #849: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 1h trend filter and session timing can capture
intraday moves while avoiding whipsaw. Key innovations:
1. 1h HMA(21) for HTF trend bias - smoother than EMA, less lag
2. 15m HMA(16) for local trend confirmation
3. RSI(7) with LOOSE thresholds (35/65) for pullback entries - ensures trades
4. Session filter: 00-12 UTC (London/NY overlap) for quality entries
5. Volume spike confirmation (>1.3x 20-bar avg) to avoid fake breakouts
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1h HMA bull + (RSI<50 OR 15m HMA bull crossover) + volume confirm
- SHORT: 1h HMA bear + (RSI>50 OR 15m HMA bear crossover) + volume confirm
- Session preference: 00-12 UTC gets full size, 12-24 UTC gets half size

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (lower for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='valid')
        # Pad with NaN at start
        return np.concatenate([np.full(span - 1, np.nan), result])
    
    close_series = pd.Series(close)
    
    # WMA(n/2)
    wma_half = wma(close, period // 2)
    # WMA(n)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n)
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=7):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20, threshold=1.3):
    """Detect volume spikes above moving average"""
    n = len(volume)
    if n < period:
        return np.zeros(n, dtype=bool)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (vol_ma * threshold)
    return spike

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    hour = pd.to_datetime(ts_seconds, unit='s').hour
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align HTF HMA
    hma_1h_raw = calculate_hma(df_1h['close'].values, 21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, 16)
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.3)
    
    # Session hours
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    prime_session = (session_hours >= 0) & (session_hours < 12)  # 00-12 UTC
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_PRIME = 0.20
    
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
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === 15m HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_15m[i-1]):
            hma_crossover_long = (close[i-1] <= hma_15m[i-1]) and (close[i] > hma_15m[i])
            hma_crossover_short = (close[i-1] >= hma_15m[i-1]) and (close[i] < hma_15m[i])
        
        # === 15m HMA TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_weak_long = rsi_7[i] < 50.0  # Loose threshold
        rsi_weak_short = rsi_7[i] > 50.0  # Loose threshold
        rsi_strong_long = rsi_7[i] < 40.0  # Stronger signal
        rsi_strong_short = rsi_7[i] > 60.0  # Stronger signal
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_spike[i]
        
        # === SESSION FILTER ===
        is_prime_session = prime_session[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        use_prime_size = False
        
        # LONG: HTF bull + (RSI weak long OR HMA crossover OR HMA bull) + volume
        if htf_1h_bull:
            entry_condition = rsi_weak_long or hma_crossover_long or hma_15m_bull
            if entry_condition:
                if vol_confirm or rsi_strong_long or hma_crossover_long:
                    # Strong signal with confirmation
                    use_prime_size = True
                    if is_prime_session:
                        desired_signal = SIZE_PRIME
                    else:
                        desired_signal = SIZE_BASE
                else:
                    # Base signal
                    if is_prime_session:
                        desired_signal = SIZE_BASE
                    else:
                        desired_signal = SIZE_BASE * 0.5  # Half size off-hours
        
        # SHORT: HTF bear + (RSI weak short OR HMA crossover OR HMA bear) + volume
        elif htf_1h_bear:
            entry_condition = rsi_weak_short or hma_crossover_short or hma_15m_bear
            if entry_condition:
                if vol_confirm or rsi_strong_short or hma_crossover_short:
                    # Strong signal with confirmation
                    use_prime_size = True
                    if is_prime_session:
                        desired_signal = -SIZE_PRIME
                    else:
                        desired_signal = -SIZE_BASE
                else:
                    # Base signal
                    if is_prime_session:
                        desired_signal = -SIZE_BASE
                    else:
                        desired_signal = -SIZE_BASE * 0.5  # Half size off-hours
        
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
        if desired_signal >= SIZE_PRIME * 0.9:
            final_signal = SIZE_PRIME
        elif desired_signal <= -SIZE_PRIME * 0.9:
            final_signal = -SIZE_PRIME
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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