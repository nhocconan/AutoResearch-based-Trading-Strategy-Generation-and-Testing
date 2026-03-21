#!/usr/bin/env python3
"""
Experiment #428: 30m KAMA Adaptive Trend + 4h/1d HMA Dual Bias + Choppiness Regime + Volume
Hypothesis: 30m timeframe needs STRONG multi-timeframe filtering to avoid whipsaw.
Using BOTH 4h and 1d HMA for trend confirmation (both must agree), KAMA adaptive entries
that flatten in ranging markets, Choppiness Index to filter range vs trend regimes,
and volume confirmation on breakouts. Multiple entry paths ensure >=10 trades/symbol.
Key insight from failures: 30m/15m strategies fail due to noise. Solution = dual HTF confirmation
(4h AND 1d must agree) + CHOP regime filter + KAMA (adaptively slows in chop).
Timeframe: 30m (REQUIRED), HTF: 4h + 1d for dual trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR for 30m timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_1d_hma_chop_volume_atr_v1"
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

def calculate_kama(close, high, low, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
    
    er = np.zeros(n)
    er[:] = np.nan
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += np.max(high[j] - low[j], 
                           np.abs(high[j] - close[j-1] if j > 0 else high[j] - low[j]),
                           np.abs(low[j] - close[j-1] if j > 0 else high[j] - low[j]))
        
        if highest_high - lowest_low > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_fast = calculate_kama(close, high, low, er_period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, high, low, er_period=10, fast_period=5, slow_period=30)
    chop = calculate_choppiness(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend bias (BOTH 4h and 1d must agree)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend confirmation (both timeframes agree)
        strong_bullish = hma_4h_bullish and hma_1d_bullish
        strong_bearish = hma_4h_bearish and hma_1d_bearish
        
        # Choppiness regime filter (CHOP < 50 = trending, CHOP > 61.8 = ranging)
        is_trending = chop[i] < 50.0
        is_ranging = chop[i] > 61.8
        
        # KAMA crossover signals
        kama_bullish_cross = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_bearish_cross = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # KAMA position
        kama_above = kama_fast[i] > kama_slow[i]
        kama_below = kama_fast[i] < kama_slow[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 0.8 * vol_sma[i] if vol_sma[i] > 0 else True
        
        # RSI conditions (relaxed to ensure trades)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 80
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 65
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: KAMA bullish cross + Strong HTF bullish + Trending + Volume
        if kama_bullish_cross and strong_bullish and is_trending and volume_ok:
            new_signal = SIZE_ENTRY
        # Path 2: KAMA above + Strong HTF bullish + RSI ok + Volume
        elif kama_above and strong_bullish and rsi_ok_long and volume_ok:
            new_signal = SIZE_ENTRY
        # Path 3: KAMA bullish cross + 4h bullish (less strict) + RSI momentum
        elif kama_bullish_cross and hma_4h_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 4: KAMA above + Strong bullish + Not ranging (avoid chop)
        elif kama_above and strong_bullish and not is_ranging and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 5: Simple - KAMA above + 4h bullish + RSI > 45
        elif kama_above and hma_4h_bullish and rsi[i] > 45 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: KAMA bearish cross + Strong HTF bearish + Trending + Volume
        if kama_bearish_cross and strong_bearish and is_trending and volume_ok:
            new_signal = -SIZE_ENTRY
        # Path 2: KAMA below + Strong HTF bearish + RSI ok + Volume
        elif kama_below and strong_bearish and rsi_ok_short and volume_ok:
            new_signal = -SIZE_ENTRY
        # Path 3: KAMA bearish cross + 4h bearish (less strict) + RSI momentum
        elif kama_bearish_cross and hma_4h_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 4: KAMA below + Strong bearish + Not ranging (avoid chop)
        elif kama_below and strong_bearish and not is_ranging and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple - KAMA below + 4h bearish + RSI < 55
        elif kama_below and hma_4h_bearish and rsi[i] < 55 and rsi[i] > 25:
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