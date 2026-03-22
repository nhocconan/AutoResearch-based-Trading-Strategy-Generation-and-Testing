#!/usr/bin/env python3
"""
Experiment #006: 1d HMA Crossover + 1w HMA Trend Filter + ADX + ATR Stop
Hypothesis: Daily timeframe captures major trends with less noise than intraday.
Weekly HMA(21) provides strong HTF trend bias - only trade in direction of weekly trend.
HMA(10)/HMA(30) crossover on daily gives smooth entry signals with less lag than EMA.
ADX(14)>18 ensures we only trade when trend has sufficient strength.
2.5*ATR(14) trailing stop appropriate for daily bars (wider than 15m).
Position size 0.30 discrete to control DD during 2022 crash.
Timeframe: 1d (REQUIRED for this experiment), HTF: 1w via mtf_data helper.
Why this might work: Daily reduces false signals, weekly filter prevents counter-trend trades
that destroyed previous strategies. HMA smoother than EMA reduces whipsaw.
Should generate 30-60 trades over 4 years (meets >=10 requirement easily).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_cross_1w_hma_adx_atr_v1"
timeframe = "1d"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Daily HMA for crossover signals
    hma_fast = calculate_hma(close, 10)
    hma_slow = calculate_hma(close, 30)
    hma_mid = calculate_hma(close, 21)
    
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
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF) - only trade in direction of weekly trend
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily HMA crossover signals
        fast_above_slow = hma_fast[i] > hma_slow[i]
        fast_below_slow = hma_fast[i] < hma_slow[i]
        
        # HMA crossover flips (entry triggers)
        cross_long = fast_above_slow and (i > 0 and hma_fast[i-1] <= hma_slow[i-1])
        cross_short = fast_below_slow and (i > 0 and hma_fast[i-1] >= hma_slow[i-1])
        
        # HMA alignment (all pointing same direction)
        hma_aligned_long = hma_fast[i] > hma_mid[i] > hma_slow[i]
        hma_aligned_short = hma_fast[i] < hma_mid[i] < hma_slow[i]
        
        # HMA slope confirmation
        hma_fast_rising = i > 0 and hma_fast[i] > hma_fast[i-1]
        hma_fast_falling = i > 0 and hma_fast[i] < hma_fast[i-1]
        
        # ADX trend strength
        trend_strong = adx[i] > 18
        trend_building = i > 0 and adx[i] > adx[i-1]
        
        # RSI filter (avoid extreme overbought/oversold entries)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        
        # Price above/below mid HMA for confirmation
        price_above_mid = close[i] > hma_mid[i]
        price_below_mid = close[i] < hma_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: HMA crossover long + weekly bullish + ADX strong + RSI ok
        if cross_long and htf_bullish and trend_strong and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # Path 2: HMA aligned long + weekly bullish + ADX building + price above mid
        elif hma_aligned_long and htf_bullish and trend_building and price_above_mid:
            new_signal = SIZE_ENTRY
        
        # Path 3: Fast HMA rising + weekly bullish + ADX strong + crossover recently happened
        elif hma_fast_rising and htf_bullish and trend_strong and fast_above_slow:
            # Check if crossover happened within last 5 days
            recent_cross = False
            for j in range(max(100, i-5), i):
                if j > 0 and hma_fast[j] > hma_slow[j] and hma_fast[j-1] <= hma_slow[j-1]:
                    recent_cross = True
                    break
            if recent_cross:
                new_signal = SIZE_ENTRY
        
        # Path 4: Weekly bullish + price pullback to mid HMA + ADX still strong
        elif htf_bullish and close[i] > hma_slow[i] and close[i] < hma_fast[i] and trend_strong:
            new_signal = SIZE_ENTRY
        
        # Path 5: Weekly bullish + HMA fast rising + ADX > 20 (strong trend continuation)
        elif htf_bullish and hma_fast_rising and adx[i] > 20 and fast_above_slow:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: HMA crossover short + weekly bearish + ADX strong + RSI ok
        if cross_short and htf_bearish and trend_strong and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # Path 2: HMA aligned short + weekly bearish + ADX building + price below mid
        elif hma_aligned_short and htf_bearish and trend_building and price_below_mid:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Fast HMA falling + weekly bearish + ADX strong + crossover recently happened
        elif hma_fast_falling and htf_bearish and trend_strong and fast_below_slow:
            # Check if crossover happened within last 5 days
            recent_cross = False
            for j in range(max(100, i-5), i):
                if j > 0 and hma_fast[j] < hma_slow[j] and hma_fast[j-1] >= hma_slow[j-1]:
                    recent_cross = True
                    break
            if recent_cross:
                new_signal = -SIZE_ENTRY
        
        # Path 4: Weekly bearish + price pullback to mid HMA + ADX still strong
        elif htf_bearish and close[i] < hma_slow[i] and close[i] > hma_fast[i] and trend_strong:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Weekly bearish + HMA fast falling + ADX > 20 (strong trend continuation)
        elif htf_bearish and hma_fast_falling and adx[i] > 20 and fast_below_slow:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe - wider stop)
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
            
            # Calculate trailing stop (2.5*ATR for daily timeframe - wider stop)
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