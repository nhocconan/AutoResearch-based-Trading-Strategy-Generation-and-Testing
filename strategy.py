#!/usr/bin/env python3
"""
Experiment #488: 30m Regime-Adaptive Fisher + Choppiness + 4h/1d HMA Trend + ATR Stop
Hypothesis: 30m timeframe needs regime detection to avoid whipsaw. Use Choppiness Index
to detect range vs trend, then apply Fisher Transform for mean-reversion entries in ranges
and HMA crossover for trend entries. 4h HMA provides intermediate trend bias, 1d HMA 
provides macro bias. Conservative sizing (0.25) with 2*ATR stoploss controls DD.
Multiple entry paths ensure >=10 trades per symbol. Must beat Sharpe=0.499 baseline.
Timeframe: 30m (REQUIRED), HTF: 4h/1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_fisher_chop_4h_1d_hma_atr_v1"
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

def calculate_fisher_transform(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals effectively in bear/range markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    signal = np.zeros(n)
    signal[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        value = 0.999 * (close[i] - lowest) / (highest - lowest) - 0.001
        value = np.clip(value, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies trending vs ranging markets.
    CHOP > 61.8 = range (mean reversion), CHOP < 38.2 = trend (trend follow).
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 50.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    fisher, fisher_signal = calculate_fisher_transform(close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # 30m HMA for trend confirmation
    hma_30m = calculate_hma(close, 21)
    hma_30m_fast = calculate_hma(close, 10)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend bias
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        hma4_bullish = close[i] > hma_4h_aligned[i]
        hma4_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m trend
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        hma_rising = hma_30m[i] > hma_30m[i-5] if i > 5 else False
        hma_falling = hma_30m[i] < hma_30m[i-5] if i > 5 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_30m_fast[i] > hma_30m[i]
        fast_below_slow = hma_30m_fast[i] < hma_30m[i]
        
        # Regime detection via Choppiness Index
        is_ranging = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # Fisher Transform signals
        fisher_cross_up = fisher[i] > -1.0 and fisher_signal[i] <= -1.0
        fisher_cross_down = fisher[i] < 1.0 and fisher_signal[i] >= 1.0
        fisher_extreme_low = fisher[i] < -1.5
        fisher_extreme_high = fisher[i] > 1.5
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = 40 < rsi[i] < 60
        
        # Price vs SMA200
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Range market + Fisher oversold cross + above SMA200
        if is_ranging and fisher_cross_up and above_sma200:
            new_signal = SIZE_ENTRY
        
        # Path 2: Trend market + 4h bullish + Daily bullish + HMA crossover up
        elif is_trending and hma4_bullish and daily_bullish and fast_above_slow:
            new_signal = SIZE_ENTRY
        
        # Path 3: Range market + RSI oversold + price > 4h HMA
        elif is_ranging and rsi_oversold and hma4_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 4: Trend market + Fisher extreme low bounce + Daily not bearish
        elif is_trending and fisher_extreme_low and fisher[i] > fisher[i-1] and not daily_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 5: HMA rising + Fast above slow + RSI neutral + 4h not bearish
        elif hma_rising and fast_above_slow and rsi_neutral and not hma4_bearish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Range market + Fisher overbought cross + below SMA200
        if is_ranging and fisher_cross_down and below_sma200:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Trend market + 4h bearish + Daily bearish + HMA crossover down
        elif is_trending and hma4_bearish and daily_bearish and fast_below_slow:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Range market + RSI overbought + price < 4h HMA
        elif is_ranging and rsi_overbought and hma4_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Trend market + Fisher extreme high drop + Daily not bullish
        elif is_trending and fisher_extreme_high and fisher[i] < fisher[i-1] and not daily_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 5: HMA falling + Fast below slow + RSI neutral + 4h not bullish
        elif hma_falling and fast_below_slow and rsi_neutral and not hma4_bullish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 30m timeframe)
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
            
            # Calculate trailing stop (2*ATR for 30m timeframe)
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