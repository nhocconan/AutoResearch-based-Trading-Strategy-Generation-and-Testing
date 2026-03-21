#!/usr/bin/env python3
"""
Hypothesis: 30m primary with 4h HTF trend + regime filter reduces whipsaw vs pure 30m strategies.
KAMA(21) adaptive trend + Bollinger BandWidth percentile regime detection avoids choppy markets.
RSI(14) pullback entries (40-60 range) in direction of 4h HMA(21) trend with ATR(14) 2.5* stoploss.
SIZE=0.30 discrete levels with regime filter should generate 20-40 trades/year with positive Sharpe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_regime_rsi_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    close_s = pd.Series(close)
    change = abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
    sc = er * (fast_sc - slow_sc) + slow_sc
    
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with bandwidth for regime detection"""
    close_s = pd.Series(close)
    sma = close_s.rolling(period, min_periods=period).mean().values
    std = close_s.rolling(period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

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
    
    # 30m indicators - all computed before loop (Rule 8)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # KAMA(21) adaptive trend
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    
    # ATR(14)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # BB Width percentile for regime (rolling 100 bars)
    bb_width_percentile = pd.Series(bb_width).rolling(100, min_periods=50).apply(
        lambda x: np.searchsorted(np.sort(x), x.iloc[-1]) / len(x), raw=False
    ).values
    bb_width_percentile = np.nan_to_num(bb_width_percentile, nan=0.5)
    
    # EMA(50) for additional trend confirmation
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # EMA(200) for long-term trend filter
    ema200 = close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start after all indicators warm up
        # HTF trend: 4h HMA direction (Rule 2 - use aligned array)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # Local trend: KAMA slope and price position
        kama_slope = kama[i] - kama[i-5] if i >= 5 else 0
        local_bullish = close[i] > kama[i] and kama_slope > 0
        local_bearish = close[i] < kama[i] and kama_slope < 0
        
        # Long-term trend filter
        lt_bullish = close[i] > ema200[i]
        lt_bearish = close[i] < ema200[i]
        
        # Regime filter: only trade in normal volatility (not extreme)
        # Avoid high vol chop (percentile > 0.75) and dead markets (percentile < 0.25)
        regime_ok = 0.25 < bb_width_percentile[i] < 0.75
        
        # RSI pullback zone (not extreme)
        rsi_neutral_long = 40 < rsi[i] < 60
        rsi_neutral_short = 40 < rsi[i] < 60
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > rsi[i-3] if i >= 3 else False
        rsi_momentum_short = rsi[i] < rsi[i-3] if i >= 3 else False
        
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
            
            # Take profit: reduce to half at 3R
            profit_r = (highest_since_entry - entry_price) / atr[i]
            if profit_r >= 3.0 and signals[i] != HALF_SIZE:
                signals[i] = HALF_SIZE
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
            
            # Take profit: reduce to half at 3R
            profit_r = (entry_price - lowest_since_entry) / atr[i]
            if profit_r >= 3.0 and signals[i] != -HALF_SIZE:
                signals[i] = -HALF_SIZE
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long: HTF bullish + local bullish + regime OK + RSI pullback
            if htf_bullish and local_bullish and regime_ok and lt_bullish:
                if rsi_neutral_long and rsi_momentum_long:
                    # Additional filter: price above BB mid
                    if close[i] > bb_mid[i]:
                        signals[i] = SIZE
                        position_side = 1
                        entry_price = close[i]
                        highest_since_entry = high[i]
            
            # Short: HTF bearish + local bearish + regime OK + RSI pullback
            elif htf_bearish and local_bearish and regime_ok and lt_bearish:
                if rsi_neutral_short and rsi_momentum_short:
                    # Additional filter: price below BB mid
                    if close[i] < bb_mid[i]:
                        signals[i] = -SIZE
                        position_side = -1
                        entry_price = close[i]
                        lowest_since_entry = low[i]
        else:
            # Hold position - maintain previous signal
            signals[i] = signals[i-1]
    
    return signals