#!/usr/bin/env python3
"""
Hypothesis: 15m primary with 4h HTF trend filter captures intraday moves while respecting macro trend.
Supertrend(10,3) for local trend + RSI(14) pullback entries + BBW regime detection.
ATR(14) stoploss at 2*ATR + trailing stop. SIZE=0.25 discrete levels minimize fee churn.
BBW percentile filters: wide bands = trend following, narrow bands = mean reversion entries.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_bbw_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator - trend direction and trailing stop"""
    n = len(close)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    return supertrend, direction, atr

def calculate_bbw_percentile(close, high, low, period=20, lookback=100):
    """Bollinger BandWidth percentile for regime detection"""
    close_s = pd.Series(close)
    sma = close_s.rolling(period, min_periods=period).mean().values
    std = close_s.rolling(period, min_periods=period).std().values
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    bbw = (bb_upper - bb_lower) / sma
    
    # Percentile of BBW over lookback period
    bbw_percentile = np.zeros(len(close))
    for i in range(lookback, len(close)):
        window = bbw[i-lookback:i+1]
        bbw_percentile[i] = np.sum(window <= bbw[i]) / len(window)
    
    return bbw, bbw_percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # 15m indicators - all computed before loop (Rule 8)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    
    # Supertrend(10, 3)
    supertrend, st_direction, atr = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # BBW regime detection
    bbw, bbw_percentile = calculate_bbw_percentile(close, high, low, period=20, lookback=100)
    
    # EMA(50) for additional trend filter
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # EMA(200) for long-term trend
    ema200 = close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start after all indicators warm up
        # HTF trend: price vs 4h HMA (Rule 2 - use aligned array)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # Local trend: Supertrend direction
        local_bullish = st_direction[i] == 1
        local_bearish = st_direction[i] == -1
        
        # Long-term trend
        lt_bullish = close[i] > ema200[i]
        lt_bearish = close[i] < ema200[i]
        
        # Regime: BBW percentile (high = trending, low = ranging)
        trending_regime = bbw_percentile[i] > 0.6
        ranging_regime = bbw_percentile[i] < 0.4
        
        # RSI conditions
        rsi_ok_long = rsi[i] < 60  # not overbought
        rsi_ok_short = rsi[i] > 40  # not oversold
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Stoploss and trailing logic (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2 * atr[i]
            initial_stop = entry_price - 2 * atr[i]
            if close[i] < max(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2 * atr[i]
            initial_stop = entry_price + 2 * atr[i]
            if close[i] > min(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long entries
            if htf_bullish and lt_bullish:
                if trending_regime and local_bullish and rsi_ok_long:
                    # Trend following in bullish regime
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                elif ranging_regime and rsi_oversold:
                    # Mean reversion in ranging regime
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                elif local_bullish and rsi_oversold:
                    # Pullback entry in uptrend
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
            
            # Short entries (more conservative)
            elif htf_bearish and lt_bearish:
                if trending_regime and local_bearish and rsi_ok_short:
                    # Trend following in bearish regime
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
                elif ranging_regime and rsi_overbought:
                    # Mean reversion in ranging regime
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
                elif local_bearish and rsi_overbought:
                    # Pullback entry in downtrend
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
        else:
            # Hold position
            signals[i] = signals[i-1]
    
    return signals