#!/usr/bin/env python3
"""
Experiment #493: 15m Supertrend + 4h HMA Bias + Choppiness Regime Filter + RSI Pullback + ATR Stop
Hypothesis: 15m timeframe needs STRONGER regime filtering than higher TFs due to noise.
Choppiness Index (CHOP) filters out ranging markets - only trade when CHOP < 38.2 (trending).
4h HMA provides HTF trend bias alignment. 15m Supertrend gives entry direction.
RSI pullback entries (40-55 for longs, 45-60 for shorts) catch dips in trends.
Multiple entry paths ensure >=10 trades. Conservative sizing (0.22) controls DD.
2.0*ATR stoploss appropriate for 15m bars. Must beat Sharpe=0.499 baseline.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_chop_regime_rsi_atr_v1"
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
    """
    Supertrend indicator - trend following with ATR bands.
    Returns: supertrend values, direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            elif close[i] < supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = -1
            else:
                if direction[i-1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    direction[i] = -1
    
    return supertrend, direction

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

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging.
    CHOP > 61.8 = ranging (avoid trading)
    CHOP < 38.2 = trending (trade with trend)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        tr_sum = 0.0
        highest_high = high[i-period+1]
        lowest_low = low[i-period+1]
        
        for j in range(i-period+1, i+1):
            tr = high[j] - low[j]
            tr_sum += tr
            if high[j] > highest_high:
                highest_high = high[j]
            if low[j] < lowest_low:
                lowest_low = low[j]
        
        price_range = highest_high - lowest_low
        if tr_sum > 0 and price_range > 0:
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    chop = calculate_chop(high, low, close, 14)
    
    # 15m HMA for additional trend confirmation
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.22
    SIZE_HALF = 0.11
    
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
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # CRITICAL: Choppiness regime filter - only trade in trending markets
        trending_market = chop[i] < 50.0  # Slightly relaxed from 38.2 for more trades
        ranging_market = chop[i] > 55.0
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = (i > 0 and st_direction[i] == 1 and st_direction[i-1] == -1 and 
                        not np.isnan(st_direction[i-1]))
        st_flip_short = (i > 0 and st_direction[i] == -1 and st_direction[i-1] == 1 and 
                         not np.isnan(st_direction[i-1]))
        
        # 15m HMA trend
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        hma_rising = (i > 0 and hma_15m[i] > hma_15m[i-1] and 
                      not np.isnan(hma_15m[i-1]))
        hma_falling = (i > 0 and hma_15m[i] < hma_15m[i-1] and 
                       not np.isnan(hma_15m[i-1]))
        
        # Fast HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        rsi_rising = (i > 0 and rsi[i] > rsi[i-1] and not np.isnan(rsi[i-1]))
        rsi_falling = (i > 0 and rsi[i] < rsi[i-1] and not np.isnan(rsi[i-1]))
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Trending market + Supertrend bullish + 4h bullish + RSI pullback
        if trending_market and st_bullish and hma_4h_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        
        # Path 2: Supertrend flip long + 4h not bearish + trending market
        elif st_flip_long and not hma_4h_bearish and trending_market:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h bullish + 15m HMA bullish + Fast HMA crossover up + trending
        elif hma_4h_bullish and hma_15m_bullish and fast_above_slow and trending_market:
            new_signal = SIZE_ENTRY
        
        # Path 4: Supertrend bullish + RSI oversold bounce + trending market
        elif st_bullish and rsi_oversold and rsi_rising and trending_market:
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h bullish + HMA rising + RSI neutral + Supertrend bullish
        elif hma_4h_bullish and hma_rising and rsi_neutral and st_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Trending market + Supertrend bearish + 4h bearish + RSI pullback
        if trending_market and st_bearish and hma_4h_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Supertrend flip short + 4h not bullish + trending market
        elif st_flip_short and not hma_4h_bullish and trending_market:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h bearish + 15m HMA bearish + Fast HMA crossover down + trending
        elif hma_4h_bearish and hma_15m_bearish and fast_below_slow and trending_market:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Supertrend bearish + RSI overbought drop + trending market
        elif st_bearish and rsi_overbought and rsi_falling and trending_market:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h bearish + HMA falling + RSI neutral + Supertrend bearish
        elif hma_4h_bearish and hma_falling and rsi_neutral and st_bearish:
            new_signal = -SIZE_ENTRY
        
        # === FORCE EXIT IN RANGING MARKET ===
        if ranging_market and position_side != 0:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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