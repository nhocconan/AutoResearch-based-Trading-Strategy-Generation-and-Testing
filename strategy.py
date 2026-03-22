#!/usr/bin/env python3
"""
Experiment #482: 30m Supertrend + 4h HMA Bias + RSI-ADX Filter + ATR Stop
Hypothesis: Supertrend(10,3) provides cleaner trend signals than KAMA/EMA crossovers.
30m timeframe captures intraday moves while 4h HMA filters out noise. Using multiple
entry paths (5 for long, 5 for short) ensures >=10 trades per symbol. RSI bands
widened to 35-65 (not extreme 20-80) for more signals. ADX threshold lowered to 15.
Position size 0.25 with 2.5*ATR stoploss for controlled drawdown.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_adx_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator - trend direction and levels."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend_dir = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    supertrend_dir[0] = -1
    
    for i in range(1, n):
        if supertrend_dir[i-1] == 1:
            # Previously bullish
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                supertrend_dir[i] = 1
            else:
                supertrend[i] = upper_band[i]
                supertrend_dir[i] = -1
        else:
            # Previously bearish
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                supertrend_dir[i] = -1
            else:
                supertrend[i] = lower_band[i]
                supertrend_dir[i] = 1
    
    return supertrend, supertrend_dir

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
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    supertrend, st_dir = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m Supertrend direction
        st_bullish = st_dir[i] == 1
        st_bearish = st_dir[i] == -1
        
        # ADX trend strength (low threshold for more trades)
        trend_strong = adx[i] > 15
        
        # RSI zones (wide bands for more trades)
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 65
        rsi_neutral_short = rsi[i] > 35 and rsi[i] < 60
        rsi_ok_long = rsi[i] > 45
        rsi_ok_short = rsi[i] < 55
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (5 paths for >=10 trades) ===
        # Path 1: 4h bullish + Supertrend long + RSI neutral
        if hma_4h_bullish and st_bullish and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 2: Supertrend long + ADX strong + RSI ok + DI bullish
        elif st_bullish and trend_strong and rsi_ok_long and di_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: 4h bullish + Supertrend long + ADX > 12
        elif hma_4h_bullish and st_bullish and adx[i] > 12:
            new_signal = SIZE_ENTRY
        # Path 4: Price above Supertrend + 4h HMA + RSI 45-60
        elif close[i] > supertrend[i] and hma_4h_bullish and rsi[i] > 45 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Path 5: Supertrend long + DI bullish + RSI > 42
        elif st_bullish and di_bullish and rsi[i] > 42:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (5 paths for >=10 trades) ===
        # Path 1: 4h bearish + Supertrend short + RSI neutral
        if hma_4h_bearish and st_bearish and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Supertrend short + ADX strong + RSI ok + DI bearish
        elif st_bearish and trend_strong and rsi_ok_short and di_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: 4h bearish + Supertrend short + ADX > 12
        elif hma_4h_bearish and st_bearish and adx[i] > 12:
            new_signal = -SIZE_ENTRY
        # Path 4: Price below Supertrend + 4h HMA + RSI 40-55
        elif close[i] < supertrend[i] and hma_4h_bearish and rsi[i] > 40 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 5: Supertrend short + DI bearish + RSI < 58
        elif st_bearish and di_bearish and rsi[i] < 58:
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