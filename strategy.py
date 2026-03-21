#!/usr/bin/env python3
"""
Hypothesis: 12h primary timeframe captures major trends with minimal noise, reducing trade count and fee drag.
KAMA(14) adaptive trend + 1d HMA(21) HTF filter + BBW regime detection for position sizing.
Asymmetric sizing: full size (0.30) in trending regimes, half size (0.15) in choppy.
ATR(14) stoploss at 3*ATR (wider for 12h) protects against whipsaws while staying in trends.
RSI(14) momentum confirmation ensures entries align with momentum direction.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bbw_regime_12h_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    close_s = pd.Series(close)
    change = abs(close_s.diff())
    volatility = close_s.diff().abs().rolling(er_period, min_periods=er_period).sum()
    er = change.rolling(er_period, min_periods=er_period).sum() / volatility
    er = er.fillna(0)
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_bbw(close, high, low, period=20):
    """Bollinger Band Width - regime detection (squeeze = low vol)"""
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    sma = close_s.rolling(period, min_periods=period).mean()
    std = close_s.rolling(period, min_periods=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bbw = (upper - lower) / sma
    return bbw.values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    hma_1d = calculate_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # 12h indicators - all computed before loop (Rule 8)
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_12h = calculate_rsi(close, 14)
    atr_12h = calculate_atr(high, low, close, 14)
    bbw_12h = calculate_bbw(close, high, low, 20)
    
    # BBW percentile for regime detection (rolling 100 bars)
    bbw_percentile = pd.Series(bbw_12h).rolling(100, min_periods=50).apply(
        lambda x: np.percentile(x, 50), raw=True
    ).values
    
    # Current BBW vs its median = regime signal
    bbw_regime = bbw_12h / bbw_percentile
    
    signals = np.zeros(n)
    SIZE_FULL = 0.30  # Full position in trending regime
    SIZE_HALF = 0.15  # Half position in choppy regime
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_size = 0.0
    
    for i in range(100, n):
        # HTF trend: price vs 1d HMA (Rule 2 - use aligned array)
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # Local trend: price vs KAMA
        local_bullish = close[i] > kama_12h[i]
        local_bearish = close[i] < kama_12h[i]
        
        # KAMA slope (trend strength)
        kama_slope = kama_12h[i] - kama_12h[i-3] if i >= 3 else 0
        kama_bullish = kama_slope > 0
        kama_bearish = kama_slope < 0
        
        # RSI momentum filter
        rsi_bullish = rsi_12h[i] > 50
        rsi_bearish = rsi_12h[i] < 50
        rsi_neutral = 40 < rsi_12h[i] < 60
        
        # Regime: trending vs choppy (BBW > median = trending)
        trending_regime = bbw_regime[i] > 1.0 if not np.isnan(bbw_regime[i]) else False
        choppy_regime = bbw_regime[i] <= 1.0 if not np.isnan(bbw_regime[i]) else True
        
        # Position size based on regime
        current_size = SIZE_FULL if trending_regime else SIZE_HALF
        
        # Stoploss and trailing logic (Rule 6) - wider stops for 12h
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 3 * atr_12h[i]
            initial_stop = entry_price - 3 * atr_12h[i]
            stop_level = max(trail_stop, initial_stop)
            
            if close[i] < stop_level:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Partial profit taking at 2R
            profit_target = entry_price + 2 * 3 * atr_12h[i]  # 2R where R=3*ATR
            if close[i] > profit_target and entry_size == SIZE_FULL:
                signals[i] = SIZE_HALF  # Reduce to half
                entry_size = SIZE_HALF
                continue
            
            # Hold position
            signals[i] = signals[i-1] if i > 0 else 0.0
        
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 3 * atr_12h[i]
            initial_stop = entry_price + 3 * atr_12h[i]
            stop_level = min(trail_stop, initial_stop)
            
            if close[i] > stop_level:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Partial profit taking at 2R
            profit_target = entry_price - 2 * 3 * atr_12h[i]
            if close[i] < profit_target and entry_size == SIZE_FULL:
                signals[i] = -SIZE_HALF
                entry_size = SIZE_HALF
                continue
            
            # Hold position
            signals[i] = signals[i-1] if i > 0 else 0.0
        
        else:
            # Flat - look for entries
            # Long: HTF bullish + KAMA bullish + RSI bullish + (trending or neutral RSI)
            long_signal = (
                htf_bullish and 
                local_bullish and 
                kama_bullish and
                rsi_bullish
            )
            
            # Short: HTF bearish + KAMA bearish + RSI bearish
            short_signal = (
                htf_bearish and 
                local_bearish and 
                kama_bearish and
                rsi_bearish
            )
            
            # Only enter in trending regime with full size, choppy with half
            if long_signal:
                signals[i] = current_size
                position_side = 1
                entry_price = close[i]
                entry_size = current_size
                highest_since_entry = high[i]
            
            elif short_signal:
                signals[i] = -current_size
                position_side = -1
                entry_price = close[i]
                entry_size = current_size
                lowest_since_entry = low[i]
    
    return signals