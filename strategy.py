#!/usr/bin/env python3
"""
Experiment #001: 15m Regime-Adaptive Strategy with Bollinger Bandwidth + RSI + Supertrend
Hypothesis: 15m timeframe captures intraday swings while 4h HMA provides trend bias.
Key innovation: Bollinger Bandwidth detects regime (squeeze=range, expand=trend).
In range: mean reversion with RSI extremes. In trend: Supertrend follow with ADX filter.
This adapts to 2022 crash (trend mode) and 2025 bear/range (mean revert mode).
Multiple entry paths ensure >=10 trades per symbol. SIZE=0.28 controls DD.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_bb_rsi_supertrend_4h_hma_v1"
timeframe = "15m"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator - trend following with ATR bands."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n) * np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            elif close[i] < supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = -1
            else:
                if direction[i-1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    direction[i] = -1
    
    return supertrend, direction

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    # Bandwidth = (Upper - Lower) / SMA
    bandwidth = np.zeros(len(close))
    for i in range(period, len(close)):
        if sma[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / sma[i]
    return upper, lower, sma, bandwidth

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_percentile_rank(values, lookback=100):
    """Calculate percentile rank for regime detection."""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(lookback, n):
        window = values[i-lookback:i]
        pr[i] = np.sum(window < values[i]) / lookback
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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_bw_percentile = calculate_percentile_rank(bb_bandwidth, 100)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(st_dir_15m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_bw_percentile[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m Supertrend direction
        st_15m_bullish = st_dir_15m[i] == 1
        st_15m_bearish = st_dir_15m[i] == -1
        
        # Supertrend flip signals on 15m
        st_flip_long = (i > 0 and st_dir_15m[i] == 1 and st_dir_15m[i-1] == -1)
        st_flip_short = (i > 0 and st_dir_15m[i] == -1 and st_dir_15m[i-1] == 1)
        
        # EMA trend
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 20
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        
        # Regime detection via Bollinger Bandwidth percentile
        # Low BW percentile = squeeze (range upcoming), High BW = expansion (trending)
        regime_range = bb_bw_percentile[i] < 0.40  # Bottom 40% = range/squeeze
        regime_trend = bb_bw_percentile[i] > 0.60  # Top 40% = trending
        
        # Bollinger position
        near_bb_lower = close[i] < bb_lower[i] * 1.005
        near_bb_upper = close[i] > bb_upper[i] * 0.995
        
        new_signal = 0.0
        
        # === LONG ENTRIES (6 paths for >=10 trades) ===
        
        # Path 1: Range regime + RSI oversold + near BB lower (mean reversion)
        if regime_range and rsi_oversold and near_bb_lower:
            new_signal = SIZE_ENTRY
        
        # Path 2: Trend regime + 4h HMA bullish + Supertrend flip long
        elif regime_trend and hma_4h_bullish and st_flip_long:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h HMA bullish + Supertrend bullish + RSI pullback
        elif hma_4h_bullish and st_15m_bullish and rsi_pullback_long and trend_strong:
            new_signal = SIZE_ENTRY
        
        # Path 4: EMA bullish + RSI oversold bounce (counter-trend in uptrend)
        elif ema_bullish and rsi_oversold and (i > 0 and rsi[i] > rsi[i-1]):
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h HMA bullish + Supertrend bullish + ADX building
        elif hma_4h_bullish and st_15m_bullish and (i > 0 and adx[i] > adx[i-1]) and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # Path 6: Range regime + price at BB lower + 4h not bearish
        elif regime_range and near_bb_lower and not hma_4h_bearish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (6 paths for >=10 trades) ===
        
        # Path 1: Range regime + RSI overbought + near BB upper (mean reversion)
        if regime_range and rsi_overbought and near_bb_upper:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Trend regime + 4h HMA bearish + Supertrend flip short
        elif regime_trend and hma_4h_bearish and st_flip_short:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h HMA bearish + Supertrend bearish + RSI pullback
        elif hma_4h_bearish and st_15m_bearish and rsi_pullback_short and trend_strong:
            new_signal = -SIZE_ENTRY
        
        # Path 4: EMA bearish + RSI overbought drop (counter-trend in downtrend)
        elif ema_bearish and rsi_overbought and (i > 0 and rsi[i] < rsi[i-1]):
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h HMA bearish + Supertrend bearish + ADX building
        elif hma_4h_bearish and st_15m_bearish and (i > 0 and adx[i] > adx[i-1]) and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # Path 6: Range regime + price at BB upper + 4h not bullish
        elif regime_range and near_bb_upper and not hma_4h_bullish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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