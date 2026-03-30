#!/usr/bin/env python3
"""
Experiment #006: 1d KAMA Momentum + RSI + Volume (1w trend)

HYPOTHESIS: KAMA(10) direction combined with RSI momentum extremes and
volume confirmation catches medium-term swings. The 1w EMA(20) filters
counter-trend trades. Simple 3-condition entry = tight enough to avoid
overtrading but loose enough to capture 60-120 trades over 4 years.

WHY IT WORKS: KAMA adapts to volatility - fast in trending markets,
slow in choppy. RSI 40/60 captures momentum without requiring extremes.
Volume confirms institutional interest. 1w trend filter avoids fighting
the major direction.

TARGET: 60-120 total trades over 4 years (15-30/year).
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(prices, period=10):
    """Kaufman's Adaptive Moving Average"""
    n = len(prices)
    if n < period:
        return np.full(n, np.nan)
    
    close = prices
    era = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        vol = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        if vol > 1e-10:
            era[i] = change / vol
        else:
            era[i] = 0.0
    
    fast_ema = np.zeros(n, dtype=np.float64)
    slow_ema = np.zeros(n, dtype=np.float64)
    kama = np.zeros(n, dtype=np.float64)
    
    fast_const = 2.0 / (2.0 + 1.0)
    slow_const = 2.0 / (30.0 + 1.0)
    
    for i in range(n):
        if i == 0:
            fast_ema[i] = close[i]
            slow_ema[i] = close[i]
        else:
            fast_ema[i] = fast_const * close[i] + (1 - fast_const) * fast_ema[i - 1]
            slow_ema[i] = slow_const * close[i] + (1 - slow_const) * slow_ema[i - 1]
        
        if era[i] > 0 and (fast_ema[i - 1] != 0 or slow_ema[i - 1] != 0):
            sc = (era[i] * (fast_const - slow_const) + slow_const) ** 2
            kama[i] = sc * close[i] + (1 - sc) * kama[i - 1] if i > 0 else close[i]
        else:
            kama[i] = close[i]
    
    return kama

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA20 for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local indicators ===
    kama = calculate_kama(close, period=10)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    
    atr_14 = calculate_atr(high, low, close, period=14)
    
    rsi = pd.Series(close).ewm(span=14, min_periods=14, adjust=False).mean()
    rsi = rsi / pd.Series(close).ewm(span=14, min_periods=14, adjust=False).mean()
    rsi = (rsi - 0.5) * 100 + 50
    rsi = pd.Series(close).apply(lambda x: 50 + (x - pd.Series(close).ewm(span=14, min_periods=14).mean().iloc[-1]) if False else 50)
    
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Proper RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1w EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if KAMA not ready
        if np.isnan(kama[i]) or np.isnan(kama_prev[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === CONDITION 1: 1w trend direction ===
        price_above_1w = close[i] > ema_1w_aligned[i]
        price_below_1w = close[i] < ema_1w_aligned[i]
        
        # === CONDITION 2: KAMA direction (rising = long bias, falling = short bias) ===
        kama_rising = kama[i] > kama_prev[i]
        kama_falling = kama[i] < kama_prev[i]
        
        # === CONDITION 3: RSI momentum (40/60 for more trades) ===
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: 1w uptrend + KAMA rising + RSI oversold + volume spike
            if price_above_1w and kama_rising and rsi_oversold and vol_spike:
                desired_signal = SIZE
            
            # SHORT: 1w downtrend + KAMA falling + RSI overbought + volume spike
            if price_below_1w and kama_falling and rsi_overbought and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (2 bars to reduce churn) ===
        bars_held = i - entry_bar
        if bars_held < 2:
            if in_position and desired_signal == 0.0:
                desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals