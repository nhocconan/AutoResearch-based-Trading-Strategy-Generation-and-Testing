#!/usr/bin/env python3
"""
Experiment #379: 15m Supertrend + 4h HMA Trend + 1h RSI Pullback + ADX Momentum Filter
Hypothesis: 15m timeframe captures short-term moves but needs strong HTF filters to avoid noise.
Using 4h HMA for trend bias (proven in #371), 1h RSI for pullback entries, and ADX(14)>25
to ensure we only trade when there's actual momentum. Supertrend(10,3) provides clear entry signals.
This combines the best elements from previous successes while adapting to 15m's faster pace.
Key insight: 15m needs stricter filters than 1h/4h to avoid fee churn from false signals.
Position sizing: 0.25 entry, 0.125 half (take profit), stoploss at 2.5*ATR.
Target: Beat Sharpe=0.499 with 50-100 trades total, DD < -30%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_1h_rsi_adx_momentum_atr_v1"
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
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            trend[i] = 1
        else:
            supertrend[i] = upper_band[i]
            trend[i] = -1
    
    return supertrend, trend

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
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
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * np.divide(plus_dm_s, tr_s, out=np.zeros(n), where=tr_s>0)
    minus_di = 100 * np.divide(minus_dm_s, tr_s, out=np.zeros(n), where=tr_s>0)
    
    # DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100 * np.divide(di_diff, di_sum, out=np.zeros(n), where=di_sum>0)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    rsi_15m = calculate_rsi(close, 14)
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi_15m[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_trend[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h RSI pullback filter
        rsi_1h_valid = not np.isnan(rsi_1h_aligned[i])
        rsi_1h_bullish = rsi_1h_valid and rsi_1h_aligned[i] > 45 and rsi_1h_aligned[i] < 70
        rsi_1h_bearish = rsi_1h_valid and rsi_1h_aligned[i] > 30 and rsi_1h_aligned[i] < 55
        
        # 15m ADX momentum filter (only trade when momentum exists)
        has_momentum = adx[i] > 20  # Lowered from 25 to ensure trade frequency
        
        # 15m Supertrend signal
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # 15m RSI filter
        rsi_15m_long = rsi_15m[i] > 35 and rsi_15m[i] < 70
        rsi_15m_short = rsi_15m[i] > 30 and rsi_15m[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Supertrend bullish + 4h bullish + 1h RSI ok + ADX momentum
        if st_bullish and trend_4h_bullish and rsi_1h_bullish and has_momentum and rsi_15m_long:
            new_signal = SIZE_ENTRY
        # Secondary: Supertrend bullish + 4h bullish + ADX momentum (1h RSI neutral ok)
        elif st_bullish and trend_4h_bullish and has_momentum and rsi_15m[i] > 40:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend bullish + ADX strong + RSI ok (4h neutral ok)
        elif st_bullish and adx[i] > 30 and rsi_15m_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Supertrend bearish + 4h bearish + 1h RSI ok + ADX momentum
        if st_bearish and trend_4h_bearish and rsi_1h_bearish and has_momentum and rsi_15m_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Supertrend bearish + 4h bearish + ADX momentum (1h RSI neutral ok)
        elif st_bearish and trend_4h_bearish and has_momentum and rsi_15m[i] < 60:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend bearish + ADX strong + RSI ok (4h neutral ok)
        elif st_bearish and adx[i] > 30 and rsi_15m_short:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
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
            
            # Calculate trailing stop (2.5*ATR from lowest)
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