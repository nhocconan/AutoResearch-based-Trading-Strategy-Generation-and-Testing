#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian breakout with weekly trend filter + RSI confirmation
- Donchian(20) captures meaningful breakouts on daily timeframe
- 1w EMA determines bull/bear regime for asymmetric entries
- RSI(14) filter avoids entering at extremes (35-65 range)
- ATR(14) trailing stop for risk management
- Fewer but higher quality trades on daily data
Timeframe: 1d (primary), 1w (HTF trend filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_rsi_regime_1d_v1"
timeframe = "1d"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average for HTF trend"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hull.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w HMA for trend filter
    hma_1w = calculate_hma(close_1w, period=10)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # 1d indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # 1d EMA for additional trend confirmation
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size - conservative for daily
    prev_signal = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend regime (1w HMA slope)
        htf_trend_bull = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        htf_trend_bear = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # 1d trend confirmation
        trend_1d_bull = ema_20[i] > ema_50[i] and close[i] > ema_50[i]
        trend_1d_bear = ema_20[i] < ema_50[i] and close[i] < ema_50[i]
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first - MUST exit on stop
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop
            if close[i] < trailing_stop or close[i] < entry_price - atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop
            if close[i] > trailing_stop or close[i] > entry_price + atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        
        # Entry logic - Donchian breakout with filters
        breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # RSI filter - avoid extremes
        rsi_ok_long = 35 < rsi[i] < 70
        rsi_ok_short = 30 < rsi[i] < 65
        
        if position_side == 0:
            # Long entry: breakout + HTF bull or neutral + RSI ok
            if breakout_long and rsi_ok_long:
                if htf_trend_bull or (not htf_trend_bear):
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    prev_signal = SIZE
                    continue
            
            # Short entry: breakout + HTF bear + RSI ok (more selective on shorts)
            if breakout_short and rsi_ok_short and htf_trend_bear and trend_1d_bear:
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = close[i]
                prev_signal = -SIZE
                continue
        
        # Exit signals - RSI extremes or trend reversal
        if position_side == 1:
            if rsi[i] > 75 or (ema_20[i] < ema_50[i] and close[i] < ema_50[i]):
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
            else:
                signals[i] = prev_signal
        elif position_side == -1:
            if rsi[i] < 25 or (ema_20[i] > ema_50[i] and close[i] > ema_50[i]):
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
            else:
                signals[i] = prev_signal
        else:
            signals[i] = prev_signal
    
    return signals