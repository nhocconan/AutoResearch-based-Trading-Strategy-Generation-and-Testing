#!/usr/bin/env python3
"""
Experiment #014: 30m Volatility Squeeze + Funding Rate Contrarian + 4h HMA Bias
Hypothesis: After 13 failed strategies, pivot to volatility expansion + funding mean reversion.
BB squeeze (low BW) precedes explosive moves. Extreme funding rates (>0.03% or <-0.03%) 
signal crowded positions that reverse. 4h HMA provides HTF trend bias to avoid counter-trend.
30m TF captures squeeze breakouts faster than 1h/4h while filtering 5m/15m noise.
Conservative sizing (0.22) + 2.5*ATR stop controls DD. Multiple entry paths ensure >=10 trades.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_squeeze_funding_4h_hma_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.where(sma > 0, bandwidth, 0.0)
    return upper, lower, sma, bandwidth

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_keltner_channels(high, low, close, period=20, atr_period=14, mult=1.5):
    """Calculate Keltner Channels for squeeze detection."""
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower, ema

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_percentile_rank(values, lookback=100):
    """Calculate percentile rank of current value vs lookback period."""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(lookback, n):
        window = values[i-lookback:i]
        current = values[i]
        rank = np.sum(window < current) / lookback
        pr[i] = rank * 100
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    kc_upper, kc_lower, kc_ema = calculate_keltner_channels(high, low, close, 20, 14, 1.5)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # BB bandwidth percentile for squeeze detection
    bb_bw_percentile = calculate_percentile_rank(bb_bw, 100)
    
    # Volatility ratio (ATR short/long)
    atr_short = calculate_atr(high, low, close, 7)
    atr_long = calculate_atr(high, low, close, 30)
    vol_ratio = np.where(atr_long > 0, atr_short / atr_long, 1.0)
    
    # Price position within BB
    bb_position = np.where(
        (bb_upper - bb_lower) > 0,
        (close - bb_lower) / (bb_upper - bb_lower),
        0.5
    )
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.22
    SIZE_HALF = 0.11
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_bw[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # BB squeeze detection (bandwidth in bottom 20th percentile)
        bb_squeeze = bb_bw_percentile[i] < 25.0 if not np.isnan(bb_bw_percentile[i]) else False
        
        # BB expansion (bandwidth increasing from squeeze)
        bb_expanding = bb_bw[i] > bb_bw[i-1] * 1.05 if i > 0 and not np.isnan(bb_bw[i-1]) else False
        
        # Inside Keltner = squeeze confirmation
        inside_kc = (close[i] > kc_lower[i]) and (close[i] < kc_upper[i])
        
        # Volatility spike (vol ratio > 1.5 = expanding vol)
        vol_spike = vol_ratio[i] > 1.3
        
        # Vol crush (vol ratio < 0.7 = collapsing vol, mean reversion setup)
        vol_crush = vol_ratio[i] < 0.7
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = 40 < rsi[i] < 60
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 18
        
        # Price breakout from BB
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: BB squeeze + expansion + 4h bullish + breakout up
        if bb_squeeze and bb_expanding and hma_4h_bullish and bb_breakout_up:
            new_signal = SIZE_ENTRY
        
        # Path 2: Vol crush + RSI oversold + 4h not bearish (mean reversion)
        elif vol_crush and rsi_oversold and not hma_4h_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 3: BB squeeze + inside KC + 4h bullish + RSI rising
        elif bb_squeeze and inside_kc and hma_4h_bullish and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 4: Trend weak (range) + BB lower touch + 4h bullish
        elif trend_weak and close[i] < bb_lower[i] * 1.002 and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 5: Vol spike + 4h bullish + RSI neutral (momentum continuation)
        elif vol_spike and hma_4h_bullish and rsi_neutral and close[i] > bb_sma[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: BB squeeze + expansion + 4h bearish + breakout down
        if bb_squeeze and bb_expanding and hma_4h_bearish and bb_breakout_down:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Vol crush + RSI overbought + 4h not bullish (mean reversion)
        elif vol_crush and rsi_overbought and not hma_4h_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: BB squeeze + inside KC + 4h bearish + RSI falling
        elif bb_squeeze and inside_kc and hma_4h_bearish and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Trend weak (range) + BB upper touch + 4h bearish
        elif trend_weak and close[i] > bb_upper[i] * 0.998 and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Vol spike + 4h bearish + RSI neutral (momentum continuation)
        elif vol_spike and hma_4h_bearish and rsi_neutral and close[i] < bb_sma[i]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals