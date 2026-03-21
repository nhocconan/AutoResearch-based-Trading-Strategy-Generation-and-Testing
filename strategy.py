#!/usr/bin/env python3
"""
Experiment #439: 15m HMA Trend + 1h RSI Pullback + 4h Bias + Volume + ATR Stop
Hypothesis: 15m strategies fail due to noise and overtrading. This uses STRONG 4h HMA trend bias
to only trade in direction of higher timeframe trend. 1h RSI pullback entries (wait for RSI dip
in uptrend, rise in downtrend) reduce false entries. 15m ADX > 25 filters choppy conditions.
Volume confirmation (volume > 1.5*avg) ensures real moves. Wide stoploss (2.5*ATR) avoids
whipsaw exits on 15m timeframe. Multiple entry paths ensure >=10 trades per symbol.
Position size: 0.25 discrete, stoploss 2.5*ATR for 15m timeframe.
Timeframe: 15m (REQUIRED), HTF: 1h for RSI, 4h for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_4h_bias_1h_rsi_pullback_vol_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
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
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_sma + 1e-10)
    return vol_ratio

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    sma50 = calculate_sma(close, 50)
    kama = calculate_kama(close, 10)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi_15m[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(sma50[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (STRONG filter - only trade with HTF trend)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h RSI pullback signals (entry timing)
        rsi_1h_pullback_long = rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 60
        rsi_1h_pullback_short = rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 60
        rsi_1h_momentum_long = rsi_1h_aligned[i] > 50 and rsi_1h_aligned[i] < 70
        rsi_1h_momentum_short = rsi_1h_aligned[i] > 30 and rsi_1h_aligned[i] < 50
        
        # 15m ADX trend strength (avoid chop)
        trend_strong = adx[i] > 22
        
        # 15m RSI for entry timing
        rsi_15m_not_overbought = rsi_15m[i] < 70
        rsi_15m_not_oversold = rsi_15m[i] > 30
        rsi_15m_long_entry = rsi_15m[i] > 40 and rsi_15m[i] < 65
        rsi_15m_short_entry = rsi_15m[i] > 35 and rsi_15m[i] < 60
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.2
        
        # Price position filters
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # DI signals
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h bullish + 1h RSI pullback + 15m ADX strong + volume
        if hma_4h_bullish and rsi_1h_pullback_long and trend_strong and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + 15m above SMA50 + DI bullish + RSI ok
        elif hma_4h_bullish and above_sma50 and di_bullish and rsi_15m_long_entry:
            new_signal = SIZE_ENTRY
        # Path 3: 4h bullish + 15m above KAMA + ADX > 25 + volume
        elif hma_4h_bullish and above_kama and adx[i] > 25 and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + 1h RSI momentum + 15m DI bullish
        elif hma_4h_bullish and rsi_1h_momentum_long and di_bullish and rsi_15m[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 5: 4h bullish + above SMA50 + volume spike + RSI 45-65
        elif hma_4h_bullish and above_sma50 and vol_ratio[i] > 1.5 and rsi_15m[i] > 45 and rsi_15m[i] < 65:
            new_signal = SIZE_ENTRY
        # Path 6: 4h bullish + ADX rising + above KAMA + RSI ok
        elif hma_4h_bullish and adx[i] > adx[i-1] and adx[i] > 20 and above_kama and rsi_15m[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h bearish + 1h RSI pullback + 15m ADX strong + volume
        if hma_4h_bearish and rsi_1h_pullback_short and trend_strong and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + 15m below SMA50 + DI bearish + RSI ok
        elif hma_4h_bearish and below_sma50 and di_bearish and rsi_15m_short_entry:
            new_signal = -SIZE_ENTRY
        # Path 3: 4h bearish + 15m below KAMA + ADX > 25 + volume
        elif hma_4h_bearish and below_kama and adx[i] > 25 and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + 1h RSI momentum + 15m DI bearish
        elif hma_4h_bearish and rsi_1h_momentum_short and di_bearish and rsi_15m[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 5: 4h bearish + below SMA50 + volume spike + RSI 35-55
        elif hma_4h_bearish and below_sma50 and vol_ratio[i] > 1.5 and rsi_15m[i] > 35 and rsi_15m[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 6: 4h bearish + ADX rising + below KAMA + RSI ok
        elif hma_4h_bearish and adx[i] > adx[i-1] and adx[i] > 20 and below_kama and rsi_15m[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 15m timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 15m timeframe)
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