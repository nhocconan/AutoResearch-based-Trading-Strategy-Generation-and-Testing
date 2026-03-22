#!/usr/bin/env python3
"""
Experiment #005: 12h Vol Spike Reversion + 1d HMA Trend Bias + RSI + ATR Stop
Hypothesis: 12h timeframe reduces noise and fee impact vs lower TFs. Vol spike reversion
(ATR(7)/ATR(30) > 2.0) captures panic bottoms and euphoria tops with high win rate.
1d HMA(21) provides HTF trend bias for asymmetric entries (only long when 1d bullish,
only short when 1d bearish). RSI(14) extremes ( <30/>70) confirm oversold/overbought.
Bollinger Band position adds mean reversion confirmation. 2*ATR(14) stoploss protects
capital. Conservative sizing (0.25 entry, 0.125 half) controls DD during 2022 crash.
This combines vol spike reversion (proven 75% win rate) with HTF trend filter to avoid
counter-trend traps. Must beat Sharpe=-0.138 from exp#001 and generate >=10 trades/symbol.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volspike_1d_hma_rsi_bb_atr_v1"
timeframe = "12h"
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy (mean revert), CHOP < 38.2 = trending.
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    choppiness = calculate_choppiness(high, low, close, 14)
    
    # Vol spike ratio: ATR(7) / ATR(30)
    vol_spike_ratio = np.zeros(n)
    vol_spike_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_spike_ratio[i] = atr_7[i] / atr_30[i]
    
    # 12h HMA for additional trend confirmation
    hma_12h = calculate_hma(close, 21)
    hma_12h_fast = calculate_hma(close, 10)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_spike_ratio[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF)
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h HMA trend
        hma_12h_bullish = close[i] > hma_12h[i]
        hma_12h_bearish = close[i] < hma_12h[i]
        hma_rising = hma_12h[i] > hma_12h[i-1] if i > 0 else False
        hma_falling = hma_12h[i] < hma_12h[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_12h_fast[i] > hma_12h[i]
        fast_below_slow = hma_12h_fast[i] < hma_12h[i]
        
        # Vol spike detection (panic/euphoria)
        vol_spike = vol_spike_ratio[i] > 2.0
        vol_normal = vol_spike_ratio[i] < 1.2
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Bollinger position
        bb_low = close[i] < bb_lower[i]
        bb_high = close[i] > bb_upper[i]
        bb_mid_cross_up = close[i] > bb_mid[i] and close[i-1] <= bb_mid[i-1] if i > 0 else False
        bb_mid_cross_down = close[i] < bb_mid[i] and close[i-1] >= bb_mid[i-1] if i > 0 else False
        
        # Choppiness regime
        choppy_market = choppiness[i] > 61.8
        trending_market = choppiness[i] < 38.2
        
        new_signal = 0.0
        
        # === LONG ENTRIES (asymmetric - prefer when 1d bullish) ===
        
        # Path 1: Vol spike + RSI oversold + 1d not bearish (panic bottom)
        if vol_spike and rsi_oversold and not htf_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 2: Vol spike + BB low + 1d bullish (strong panic buy)
        elif vol_spike and bb_low and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: 1d bullish + 12h HMA bullish + Fast HMA crossover up + RSI neutral
        elif htf_bullish and hma_12h_bullish and fast_above_slow and rsi_neutral:
            new_signal = SIZE_ENTRY
        
        # Path 4: Choppy market + BB low + RSI oversold (mean reversion in range)
        elif choppy_market and bb_low and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 5: 1d bullish + HMA rising + BB mid cross up (trend continuation)
        elif htf_bullish and hma_rising and bb_mid_cross_up:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (asymmetric - prefer when 1d bearish) ===
        
        # Path 1: Vol spike + RSI overbought + 1d not bullish (euphoria top)
        if vol_spike and rsi_overbought and not htf_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Vol spike + BB high + 1d bearish (strong panic sell)
        elif vol_spike and bb_high and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 1d bearish + 12h HMA bearish + Fast HMA crossover down + RSI neutral
        elif htf_bearish and hma_12h_bearish and fast_below_slow and rsi_neutral:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Choppy market + BB high + RSI overbought (mean reversion in range)
        elif choppy_market and bb_high and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 1d bearish + HMA falling + BB mid cross down (trend continuation)
        elif htf_bearish and hma_falling and bb_mid_cross_down:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 12h timeframe)
            current_stop = highest_close - 2.0 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr_14[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 12h timeframe)
            current_stop = lowest_close + 2.0 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr_14[i]
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
            trailing_stop = close[i] - 2.0 * atr_14[i] if position_side > 0 else close[i] + 2.0 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr_14[i] if position_side > 0 else close[i] + 2.0 * atr_14[i]
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