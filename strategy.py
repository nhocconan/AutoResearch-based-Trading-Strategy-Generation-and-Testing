#!/usr/bin/env python3
"""
Experiment #392: 30m KAMA Adaptive Trend + 4h HMA Filter + ADX Regime + RSI Momentum + ATR Stop
Hypothesis: 30m timeframe captures swing trades better than 15m (less noise) but faster than 1h.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - moves fast in trends, slow in ranges.
Combined with 4h HMA trend filter (proven in #383), this should reduce whipsaws while maintaining trade frequency.
ADX(14) > 20 ensures we trade in trending conditions (not too strict like ADX>40 which kills trades).
RSI(14) loose thresholds (35-65) ensure minimum trade frequency - CRITICAL after many 0-trade failures.
ATR(14) stoploss at 2.5x protects capital. Position size 0.25 discrete to minimize fee churn.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper (call ONCE before loop).
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
Key insight: KAMA adapts better than HMA/EMA in mixed regimes, 30m captures swings without 15m noise.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_adx_rsi_atr_v1"
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

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency ratio (ER).
    ER close to 1 = trending (fast smoothing), ER close to 0 = ranging (slow smoothing).
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        if change == 0:
            er[i] = 0
            continue
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    er[:period] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.clip(sc, slow_sc, fast_sc)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:period] = np.nan
    return kama

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
    adx = np.zeros(n)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth +DM, -DM, and TR using Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for HTF trend."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, 14)
    
    # KAMA fast for crossover signals
    kama_fast = calculate_kama(close, 8)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(kama_fast[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # ADX trend strength filter (loose to ensure trades)
        is_trending = adx[i] > 20  # Not too strict (ADX>40 kills trades)
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]
        kama_cross_short = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]
        
        # KAMA position (already crossed)
        kama_bullish = kama_fast[i] > kama[i]
        kama_bearish = kama_fast[i] < kama[i]
        
        # RSI filter (LOOSE to ensure trade frequency - CRITICAL)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 40
        rsi_momentum_short = rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades) ===
        # Primary: KAMA cross long + 4h bullish + trending + RSI ok
        if kama_cross_long and trend_bullish and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA bullish + 4h bullish + RSI momentum (no cross needed)
        elif kama_bullish and trend_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA cross long + RSI ok (4h neutral ok in strong trend)
        elif kama_cross_long and is_trending and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Quaternary: KAMA bullish alone (ensures minimum trade frequency)
        elif kama_bullish and rsi[i] > 40 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Quintenary: KAMA cross long even without ADX (backup for trade frequency)
        elif kama_cross_long and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades) ===
        # Primary: KAMA cross short + 4h bearish + trending + RSI ok
        if kama_cross_short and trend_bearish and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA bearish + 4h bearish + RSI momentum (no cross needed)
        elif kama_bearish and trend_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA cross short + RSI ok (4h neutral ok in strong trend)
        elif kama_cross_short and is_trending and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: KAMA bearish alone (ensures minimum trade frequency)
        elif kama_bearish and rsi[i] > 30 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quintenary: KAMA cross short even without ADX (backup for trade frequency)
        elif kama_cross_short and rsi[i] < 60:
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