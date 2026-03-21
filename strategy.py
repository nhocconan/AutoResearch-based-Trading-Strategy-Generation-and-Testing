#!/usr/bin/env python3
"""
Hypothesis: 1h primary with 12h HTF trend filter + Bollinger BandWidth regime detection.
Donchian(20) breakouts only when BBW percentile < 30 (squeeze before expansion).
ADX(14) > 25 filter ensures trending conditions, avoiding choppy whipsaws.
ATR(14) stoploss at 2.5*ATR + trailing stop protects capital.
SIZE=0.30 discrete levels with fewer trades to minimize fee churn.
Key insight: Failed strategies had 3000+ trades (fee death). Target 50-100 trades/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_bbw_adx_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    prev_close = close_s.shift(1).fillna(close[0])
    tr = np.maximum(high_s - low_s, np.maximum(abs(high_s - prev_close), abs(low_s - prev_close)))
    
    # Directional Movement
    plus_dm = np.where((high_s - high_s.shift(1)) > (low_s.shift(1) - low_s), 
                       np.maximum(high_s - high_s.shift(1), 0), 0)
    minus_dm = np.where((low_s.shift(1) - low_s) > (high_s - high_s.shift(1)), 
                        np.maximum(low_s.shift(1) - low_s, 0), 0)
    
    # Smoothed averages
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_bbw(high, low, close, period=20):
    """Bollinger Band Width - volatility measure"""
    close_s = pd.Series(close)
    sma = close_s.rolling(period, min_periods=period).mean().values
    std = close_s.rolling(period, min_periods=period).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    bbw = (upper - lower) / sma
    return bbw, upper, lower, sma

def calculate_bbw_percentile(bbw, lookback=100):
    """Percentile rank of current BBW vs recent history"""
    bbw_s = pd.Series(bbw)
    percentile = bbw_s.rolling(lookback, min_periods=lookback).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x), raw=False
    ).values
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # 1h indicators - all computed before loop (Rule 8)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # ATR(14)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian(20)
    donchian_upper = pd.Series(high).rolling(20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # EMA(50) for local trend
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # ADX(14) for trend strength
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Bollinger Band Width and percentile
    bbw, bb_upper, bb_lower, bb_sma = calculate_bbw(high, low, close, 20)
    bbw_pct = calculate_bbw_percentile(bbw, 100)
    
    # RSI(14) for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    
    signals = np.zeros(n)
    SIZE = 0.30
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(adx[i]) or np.isnan(bbw_pct[i]):
            continue
        
        # HTF trend: 12h HMA direction
        htf_bullish = close[i] > hma_12h_aligned[i]
        htf_bearish = close[i] < hma_12h_aligned[i]
        
        # Local trend: EMA50
        local_bullish = close[i] > ema50[i] and ema50[i] > ema50[i-1] if i > 0 else False
        local_bearish = close[i] < ema50[i] and ema50[i] < ema50[i-1] if i > 0 else False
        
        # Regime: BBW percentile < 30 = squeeze (volatility expansion coming)
        squeeze = bbw_pct[i] < 0.30
        
        # Trend strength: ADX > 25 = trending market
        trending = adx[i] > 25
        
        # Donchian breakout (use previous bar to avoid look-ahead)
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # RSI filter - avoid extreme entries
        rsi_ok_long = rsi[i] < 70
        rsi_ok_short = rsi[i] > 30
        
        # Stoploss and trailing logic (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2.5 * atr[i]
            initial_stop = entry_price - 2.5 * atr[i]
            stop_level = max(trail_stop, initial_stop)
            if close[i] < stop_level:
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            initial_stop = entry_price + 2.5 * atr[i]
            stop_level = min(trail_stop, initial_stop)
            if close[i] > stop_level:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long: HTF bullish + local bullish + squeeze + trending + breakout
            if htf_bullish and local_bullish:
                if breakout_long and rsi_ok_long and (squeeze or trending):
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
            
            # Short: HTF bearish + local bearish + squeeze + trending + breakout
            elif htf_bearish and local_bearish:
                if breakout_short and rsi_ok_short and (squeeze or trending):
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
        else:
            # Hold position - maintain signal
            signals[i] = signals[i-1]
    
    return signals