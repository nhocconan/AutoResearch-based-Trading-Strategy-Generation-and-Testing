#!/usr/bin/env python3
"""
Experiment #008: 30m Fisher Transform + 4h HMA Trend + BB Regime + ATR Stop
Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(like 2025+). Combined with 4h HMA trend bias and Bollinger Band width regime filter,
this should generate quality entries while avoiding whipsaws. Fisher crossings at
extremes (-1.5/+1.5) signal reversals, 4h HMA provides directional bias, BB width
detects squeeze vs expansion regimes. Conservative sizing (0.25) with 2.5*ATR stop.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_bb_regime_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to -1 to +1 range.
    Crossings at extremes signal reversals. Excellent for bear/range markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        x = (hl2[i] - lowest) / (highest - lowest)
        
        # Constrain to 0.001-0.999 to avoid log(0)
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fish = 0.5 * np.log((1 + x) / (1 - x))
        
        # Smooth with previous value (Ehlers method)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fish + 0.33 * fisher[i-1]
        else:
            fisher[i] = fish
    
    return fisher

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma * 100.0
    return upper, lower, sma, bandwidth

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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 30m HMA for additional confirmation
    hma_30m = calculate_hma(close, 21)
    hma_30m_fast = calculate_hma(close, 10)
    
    # Calculate BB bandwidth percentile for regime detection
    bb_bw_percentile = np.zeros(n)
    bb_bw_percentile[:] = np.nan
    lookback = 100
    for i in range(lookback, n):
        if np.isnan(bb_bandwidth[i]):
            continue
        valid_bw = bb_bandwidth[i-lookback:i+1]
        valid_bw = valid_bw[~np.isnan(valid_bw)]
        if len(valid_bw) > 0:
            bb_bw_percentile[i] = np.sum(bb_bandwidth[i] > valid_bw) / len(valid_bw) * 100.0
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m HMA trend
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        hma_rising = hma_30m[i] > hma_30m[i-1] if i > 0 else False
        hma_falling = hma_30m[i] < hma_30m[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_30m_fast[i] > hma_30m[i]
        fast_below_slow = hma_30m_fast[i] < hma_30m[i]
        
        # Fisher Transform signals
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # BB regime
        bb_squeeze = bb_bw_percentile[i] < 30.0 if not np.isnan(bb_bw_percentile[i]) else False
        bb_expansion = bb_bw_percentile[i] > 70.0 if not np.isnan(bb_bw_percentile[i]) else False
        price_near_lower = close[i] < bb_lower[i] * 1.002
        price_near_upper = close[i] > bb_upper[i] * 0.998
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        
        # Z-score mean reversion
        zscore_extreme_long = zscore[i] < -1.5
        zscore_extreme_short = zscore[i] > 1.5
        
        # Volume confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Fisher cross up + 4h bullish + RSI pullback
        if fisher_cross_up and trend_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        
        # Path 2: Fisher oversold + 4h bullish + BB squeeze (mean revert in trend)
        elif fisher_oversold and trend_bullish and bb_squeeze:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h bullish + Fast HMA crossover + Volume spike
        elif trend_bullish and fast_above_slow and volume_spike and hma_30m_fast[i] > hma_30m_fast[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 4: Z-score extreme + 4h bullish + RSI oversold bounce
        elif zscore_extreme_long and trend_bullish and rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: Price at BB lower + 4h bullish + Fisher turning up
        elif price_near_lower and trend_bullish and fisher[i] > fisher[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Fisher cross down + 4h bearish + RSI pullback
        if fisher_cross_down and trend_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Fisher overbought + 4h bearish + BB squeeze (mean revert in trend)
        elif fisher_overbought and trend_bearish and bb_squeeze:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h bearish + Fast HMA crossover down + Volume spike
        elif trend_bearish and fast_below_slow and volume_spike and hma_30m_fast[i] < hma_30m_fast[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Z-score extreme + 4h bearish + RSI overbought drop
        elif zscore_extreme_short and trend_bearish and rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Price at BB upper + 4h bearish + Fisher turning down
        elif price_near_upper and trend_bearish and fisher[i] < fisher[i-1] if i > 0 else False:
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