#!/usr/bin/env python3
"""
Experiment #409: 15m Mean Reversion + 4h HMA Trend Bias + Bollinger Regime + ATR Stop
Hypothesis: 15m timeframe is too noisy for pure trend following (see #397-#405 failures).
Mean reversion with HTF trend bias should work better: enter on RSI extremes when aligned
with 4h trend direction. Bollinger Band squeeze detection filters low-volatility periods
where mean reversion fails. Volume confirmation reduces false signals. Target: Beat Sharpe=0.499
with >=10 trades/symbol by using multiple entry paths and relaxed RSI thresholds.
Timeframe: 15m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2*ATR for 15m timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_mr_4h_hma_bb_regime_volume_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma  # Bandwidth
    pct_b = (close - lower) / (upper - lower)  # %B indicator
    return upper, lower, bw, pct_b

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / std
    return zscore

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bw, bb_pct_b = calculate_bollinger_bands(close, 20, 2.0)
    vol_ma = calculate_volume_ma(volume, 20)
    zscore = calculate_zscore(close, 20)
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):  # Start after 200 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_bw[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma50[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if atr[i] == 0 or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Bollinger Band regime (squeeze = low vol, expansion = high vol)
        bb_squeeze = bb_bw[i] < np.nanpercentile(bb_bw[:i], 25) if i > 20 else False
        bb_expansion = bb_bw[i] > np.nanpercentile(bb_bw[:i], 75) if i > 20 else False
        
        # Volume confirmation (above average = real move)
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # RSI mean reversion signals
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        # Z-score mean reversion
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # %B indicator (position within bands)
        pct_b_low = bb_pct_b[i] < 0.1
        pct_b_high = bb_pct_b[i] > 0.9
        
        # Price position relative to SMA50
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (mean reversion with HTF trend bias) ===
        # Path 1: RSI oversold + 4h bullish trend + volume confirmed
        if rsi_oversold and trend_bullish and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 2: RSI extreme oversold + 4h bullish (volume optional for extreme)
        elif rsi_extreme_oversold and trend_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: Z-score oversold + RSI oversold + 4h bullish
        elif zscore_oversold and rsi_oversold and trend_bullish:
            new_signal = SIZE_ENTRY
        # Path 4: %B low + RSI oversold + above SMA50 (pullback in uptrend)
        elif pct_b_low and rsi_oversold and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 5: BB squeeze breakout long + 4h bullish
        elif bb_expansion and close[i] > bb_upper[i] and trend_bullish and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 6: Simple RSI mean reversion + 4h bullish bias
        elif rsi[i] < 40 and trend_bullish and above_sma50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (mean reversion with HTF trend bias) ===
        # Path 1: RSI overbought + 4h bearish trend + volume confirmed
        if rsi_overbought and trend_bearish and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 2: RSI extreme overbought + 4h bearish (volume optional for extreme)
        elif rsi_extreme_overbought and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: Z-score overbought + RSI overbought + 4h bearish
        elif zscore_overbought and rsi_overbought and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Path 4: %B high + RSI overbought + below SMA50 (rally in downtrend)
        elif pct_b_high and rsi_overbought and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 5: BB squeeze breakout short + 4h bearish
        elif bb_expansion and close[i] < bb_lower[i] and trend_bearish and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 6: Simple RSI mean reversion + 4h bearish bias
        elif rsi[i] > 60 and trend_bearish and below_sma50:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest for 15m timeframe)
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
            
            # Calculate trailing stop (2*ATR from lowest for 15m timeframe)
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
        prev_signal = signals[i-1] if i > 0 else 0.0
        
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