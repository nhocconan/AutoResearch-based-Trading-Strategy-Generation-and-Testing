#!/usr/bin/env python3
"""
Experiment #007: 15m Primary Timeframe Strategy
Hypothesis: 4h HMA trend filter + 15m RSI pullback entries + ATR stoploss
- Use 4h HMA(21) for trend direction (HTF)
- Use 15m RSI(14) for entry timing on pullbacks
- ATR(14) trailing stop at 2*ATR
- Discrete position sizing: 0.0, ±0.25, ±0.125 (half at take profit)
- Volume confirmation to filter false breakouts
Why this should work: MTF alignment proven in baseline, RSI pullbacks catch dips in trends
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_atr_15m_v2"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average calculation"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2 * wma1 - wma2
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range calculation"""
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI calculation"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_volume_ma(volume, period=20):
    """Volume moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate LTF indicators (15m)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # EMA for additional trend confirmation on LTF
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital)
    HALF_SIZE = 0.125  # Half position for take profit
    
    # Track position state for stoploss/take profit
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    prev_signal = 0.0
    
    for i in range(100, n):
        # HTF trend from 4h HMA
        hma_trend = 1.0 if hma_4h_aligned[i] > hma_4h_aligned[i-1] else -1.0
        
        # LTF trend from EMA
        ltf_trend = 1.0 if ema21[i] > ema50[i] else -1.0
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * 0.8  # At least 80% of avg volume
        
        # ATR percentage for stoploss calculation
        atr_pct = atr[i] / close[i] if close[i] > 0 else 0.02
        
        # Initialize tracking arrays
        if i == 100:
            entry_price[i] = close[i]
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        else:
            entry_price[i] = entry_price[i-1]
            highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
            lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
        
        current_signal = 0.0
        
        # LONG entry conditions
        if hma_trend > 0 and ltf_trend > 0 and vol_confirmed:
            # RSI pullback entry (buy dip in uptrend)
            if rsi[i] >= 40 and rsi[i] <= 60:
                current_signal = SIZE
                if prev_signal <= 0:
                    entry_price[i] = close[i]
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
            # RSI strong momentum
            elif rsi[i] > 60 and rsi[i] < 75:
                current_signal = SIZE
                if prev_signal <= 0:
                    entry_price[i] = close[i]
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
        
        # SHORT entry conditions
        elif hma_trend < 0 and ltf_trend < 0 and vol_confirmed:
            # RSI pullback entry (sell rally in downtrend)
            if rsi[i] >= 40 and rsi[i] <= 60:
                current_signal = -SIZE
                if prev_signal >= 0:
                    entry_price[i] = close[i]
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
            # RSI strong momentum
            elif rsi[i] > 25 and rsi[i] < 40:
                current_signal = -SIZE
                if prev_signal >= 0:
                    entry_price[i] = close[i]
                    highest_since_entry[i] = close[i]
                    lowest_since_entry[i] = close[i]
        
        # Stoploss logic (2*ATR)
        if prev_signal > 0:  # Long position
            stoploss_price = entry_price[i] - 2.0 * atr[i]
            if close[i] < stoploss_price:
                current_signal = 0.0  # Stoploss hit
            # Take profit at 2R (reduce to half)
            elif close[i] > entry_price[i] + 4.0 * atr[i]:
                current_signal = HALF_SIZE
            # Trail stop at 1R profit
            elif close[i] > entry_price[i] + 2.0 * atr[i]:
                if current_signal > 0:
                    pass  # Hold position
        
        if prev_signal < 0:  # Short position
            stoploss_price = entry_price[i] + 2.0 * atr[i]
            if close[i] > stoploss_price:
                current_signal = 0.0  # Stoploss hit
            # Take profit at 2R (reduce to half)
            elif close[i] < entry_price[i] - 4.0 * atr[i]:
                current_signal = -HALF_SIZE
            # Trail stop at 1R profit
            elif close[i] < entry_price[i] - 2.0 * atr[i]:
                if current_signal < 0:
                    pass  # Hold position
        
        # Update tracking
        position_side[i] = np.sign(current_signal)
        if current_signal != prev_signal and current_signal != 0:
            entry_price[i] = close[i]
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        signals[i] = current_signal
        prev_signal = current_signal
    
    return signals