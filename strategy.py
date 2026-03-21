#!/usr/bin/env python3
"""
Hypothesis: 30m primary with 4h HMA trend filter + RSI pullback entries + ATR stoploss.
The 4h trend provides directional bias, 30m RSI catches pullbacks in trend direction,
and ATR-based stops limit drawdown. Discrete signal levels reduce fee churn.
Timeframe: 30m (required for this experiment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_atr_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average for trend detection."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hma = (2*wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI for pullback detection."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_adx(high, low, close, period=14):
    """ADX for trend strength filter."""
    plus_dm = np.where(high - np.roll(high, 1) > np.roll(low, 1) - low, 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where(np.roll(low, 1) - low > high - np.roll(high, 1),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # 30m HMA for short-term trend
    hma_30m = calculate_hma(close, 16)
    
    signals = np.zeros(n)
    SIZE_NORMAL = 0.25
    SIZE_STRONG = 0.35
    
    # Track position for stoploss
    entry_price = 0.0
    position_side = 0
    
    for i in range(100, n):
        # HTF trend bias from 4h HMA
        hma_4h_val = hma_4h_aligned[i]
        hma_4h_prev = hma_4h_aligned[i-1] if i > 0 else hma_4h_val
        htf_trend = 1.0 if hma_4h_val > hma_4h_prev else -1.0
        
        # ADX trend strength filter
        adx_val = adx[i]
        trend_strong = adx_val > 20
        
        # RSI pullback levels
        rsi_val = rsi[i]
        
        # Short-term trend from 30m HMA
        hma_30m_val = hma_30m[i]
        hma_30m_prev = hma_30m[i-1] if i > 0 else hma_30m_val
        stf_trend = 1.0 if hma_30m_val > hma_30m_prev else -1.0
        
        # Stoploss check (2*ATR)
        if position_side == 1 and close[i] < entry_price - 2.0 * atr[i]:
            signals[i] = 0.0
            position_side = 0
            continue
        elif position_side == -1 and close[i] > entry_price + 2.0 * atr[i]:
            signals[i] = 0.0
            position_side = 0
            continue
        
        # Entry logic: HTF trend + pullback + ADX confirmation
        if htf_trend > 0 and trend_strong:
            # Long on RSI pullback (40-55 range)
            if 40 <= rsi_val <= 55 and stf_trend > 0:
                signals[i] = SIZE_STRONG if adx_val > 30 else SIZE_NORMAL
                if position_side <= 0:
                    entry_price = close[i]
                    position_side = 1
            elif rsi_val > 65:
                # Overbought - reduce or exit
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
                
        elif htf_trend < 0 and trend_strong:
            # Short on RSI bounce (45-60 range)
            if 45 <= rsi_val <= 60 and stf_trend < 0:
                signals[i] = -SIZE_STRONG if adx_val > 30 else -SIZE_NORMAL
                if position_side >= 0:
                    entry_price = close[i]
                    position_side = -1
            elif rsi_val < 35:
                # Oversold - reduce or exit
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        else:
            # Weak trend - stay flat or reduce
            signals[i] = 0.0
            position_side = 0
    
    return signals