#!/usr/bin/env python3
"""
Experiment #416: 30m RSI Pullback + 4h HMA Trend + Z-Score Mean Reversion + ATR Stop
Hypothesis: 30m timeframe captures swing trades better than 15m (less noise) and 1h (more signals).
Combined with 4h HMA for trend bias and Z-score for mean reversion confirmation, this should
generate sufficient trades (>=10/symbol) while maintaining positive Sharpe. Key insight from
failures: Supertrend and Fisher on shorter TFs failed badly. RSI pullback with loose thresholds
works better for generating trades. Multiple entry paths ensure we don't get 0 trades.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Position size: 0.25-0.30 discrete, stoploss 2.5*ATR for 30m timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_hma_zscore_atr_v1"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    return zscore.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    sma50 = calculate_sma(close, 50)
    ema21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma50[i]) or np.isnan(ema21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (long-term direction) - SOFT filter
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        above_ema21 = close[i] > ema21[i]
        below_ema21 = close[i] < ema21[i]
        
        # RSI levels (LOOSE thresholds to ensure trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 65
        
        # Z-score mean reversion
        zscore_oversold = zscore[i] < -1.0
        zscore_overbought = zscore[i] > 1.0
        zscore_extreme_long = zscore[i] < -1.5
        zscore_extreme_short = zscore[i] > 1.5
        
        # RSI momentum
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h bullish + RSI pullback + Z-score oversold (primary)
        if trend_4h_bullish and rsi_oversold and zscore_oversold:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + RSI rising + above EMA21
        elif trend_4h_bullish and rsi_rising and above_ema21 and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 3: Above SMA50 + RSI neutral + Z-score extreme
        elif above_sma50 and rsi_neutral_long and zscore_extreme_long:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + RSI crossing up from oversold
        elif trend_4h_bullish and rsi[i] > 40 and rsi[i-1] <= 40 and rsi_rising:
            new_signal = SIZE_ENTRY
        # Path 5: Simple momentum - 4h bullish + above EMA21 + RSI > 50
        elif trend_4h_bullish and above_ema21 and rsi[i] > 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h bearish + RSI overbought + Z-score overbought (primary)
        if trend_4h_bearish and rsi_overbought and zscore_overbought:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + RSI falling + below EMA21
        elif trend_4h_bearish and rsi_falling and below_ema21 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 3: Below SMA50 + RSI neutral + Z-score extreme
        elif below_sma50 and rsi_neutral_short and zscore_extreme_short:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + RSI crossing down from overbought
        elif trend_4h_bearish and rsi[i] < 60 and rsi[i-1] >= 60 and rsi_falling:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple momentum - 4h bearish + below EMA21 + RSI < 50
        elif trend_4h_bearish and below_ema21 and rsi[i] < 50:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 30m timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 30m timeframe)
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
        prev_signal = signals[i-1] if i > 0 else 0.0
        
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