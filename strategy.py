#!/usr/bin/env python3
"""
Experiment #478: 4h KAMA Adaptive Trend + Daily HMA Bias + Volume Breakout + ATR Stop

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adjusts speed based on market efficiency.
In trending markets (high ER), KAMA moves fast and follows price. In choppy markets (low ER),
KAMA flattens and reduces whipsaws. Combined with ADX>20 trend filter, volume confirmation
(>1.3x 20-bar avg), and Daily HMA bias, this should capture trends while avoiding chop.

Key innovation vs failed strategies:
- KAMA instead of HMA/EMA (adaptive to market conditions)
- Volume confirmation (most failed 4h strategies ignored volume)
- Simpler entry logic (3 clear paths instead of 5+ complex filters)
- Proper stoploss tracking (2.5*ATR trailing)

Timeframe: 4h (REQUIRED per experiment #478)
HTF: 1d via mtf_data.get_htf_data() - called ONCE before loop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_volume_breakout_atr_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market volatility via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER = trending (KAMA moves fast), Low ER = choppy (KAMA flat)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < er_period + 5:
        return kama
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close - np.roll(close, er_period))
    net_change[:er_period] = np.nan
    
    sum_changes = np.zeros(n)
    for i in range(er_period, n):
        sum_changes[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = net_change / (sum_changes + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constants
    fast_sc = (2 / (fast_period + 1)) ** 2
    slow_sc = (2 / (slow_period + 1)) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength measurement."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 3:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.where(atr > 0, plus_di / (atr + 1e-10), 0)
    minus_di = np.where(atr > 0, minus_di / (atr + 1e-10), 0)
    
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10), 0)
    adx[period*2:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period*2:]
    
    return adx

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

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_ma = pd.Series(volume).ewm(span=period, min_periods=period, adjust=False).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_slope = calculate_slope(kama, lookback=5)
    vol_ma = calculate_volume_ma(volume, 20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(kama_slope[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # ADX trend strength filter
        adx_trending = adx[i] > 20
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # RSI neutral zone (not overbought/oversold)
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # KAMA crossover detection
        kama_cross_up = False
        kama_cross_down = False
        if i > 0 and not np.isnan(kama[i-1]):
            if close[i-1] <= kama[i-1] and close[i] > kama[i]:
                kama_cross_up = True
            elif close[i-1] >= kama[i-1] and close[i] < kama[i]:
                kama_cross_down = True
        
        new_signal = 0.0
        
        # === LONG ENTRIES (3 paths for sufficient trades) ===
        # Path 1: Daily bullish + 4h KAMA bullish + ADX trending + Volume confirmed
        if daily_bullish and kama_bullish and adx_trending and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 2: Daily bullish + KAMA cross up + RSI neutral
        elif daily_bullish and kama_cross_up and rsi_neutral:
            new_signal = SIZE_ENTRY
        # Path 3: 4h KAMA bullish + KAMA rising + Volume confirmed + RSI > 45
        elif kama_bullish and kama_rising and volume_confirmed and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (3 paths for sufficient trades) ===
        # Path 1: Daily bearish + 4h KAMA bearish + ADX trending + Volume confirmed
        if daily_bearish and kama_bearish and adx_trending and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 2: Daily bearish + KAMA cross down + RSI neutral
        elif daily_bearish and kama_cross_down and rsi_neutral:
            new_signal = -SIZE_ENTRY
        # Path 3: 4h KAMA bearish + KAMA falling + Volume confirmed + RSI < 55
        elif kama_bearish and kama_falling and volume_confirmed and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[entry_idx] if 'entry_idx' in dir() else 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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
            entry_idx = i
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
            entry_idx = i
        
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values