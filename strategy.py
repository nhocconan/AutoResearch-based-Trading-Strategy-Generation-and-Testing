#!/usr/bin/env python3
"""
Experiment #003: 1h Connors RSI + Bollinger Mean Reversion + 4h HMA Regime + Vol Spike Filter
Hypothesis: Simple trend-following (Supertrend/EMA) failed on BTC/ETH because 2025 is bear/range.
Mean reversion strategies work better in choppy markets. Connors RSI (CRSI) has 75% win rate
for short-term reversals. Combined with Bollinger Band extremes and vol spike filter
(ATR(7)/ATR(30) > 1.8), we catch panic reversals. 4h HMA provides regime bias - only take
longs when 4h bullish, shorts when 4h bearish. Multiple entry paths ensure >=10 trades.
Conservative sizing (0.25) with 2*ATR stoploss controls DD during 2022 crash.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_bb_4h_hma_volspike_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Measures short-term momentum and mean reversion potential.
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI(2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        gains = np.sum(np.where(streak_vals > 0, streak_vals, 0))
        losses = np.abs(np.sum(np.where(streak_vals < 0, streak_vals, 0)))
        if losses > 0:
            streak_rsi[i] = 100 - 100 / (1 + gains / losses)
        else:
            streak_rsi[i] = 100.0
    
    # Percent Rank(100) - where current price ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_short[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bw = (upper - lower) / sma  # Bandwidth
    return upper, lower, sma, bw

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies ranging vs trending markets.
    CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(calculate_atr(high[i-period+1:i+1], low[i-period+1:i+1], close[i-period+1:i+1], 1))
        
        if highest > lowest and atr_sum > 0:
            chop[i] = 100 * np.log10((highest - lowest) / atr_sum) / np.log10(period)
    
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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    
    bb_upper, bb_lower, bb_mid, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    hma_1h = calculate_hma(close, 21)
    
    chop = calculate_choppiness(high, low, close, 14)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h regime bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # Volatility spike filter (ATR ratio)
        vol_spike = (atr_7[i] / atr_30[i]) > 1.8 if atr_30[i] > 0 else False
        vol_normal = (atr_7[i] / atr_30[i]) < 1.3 if atr_30[i] > 0 else True
        
        # Choppiness regime
        is_ranging = chop[i] > 55
        is_trending = chop[i] < 45
        
        # Connors RSI extremes
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_neutral = crsi[i] > 30 and crsi[i] < 70
        
        # Bollinger position
        bb_low = close[i] < bb_lower[i]
        bb_high = close[i] > bb_upper[i]
        bb_mid_cross_up = close[i] > bb_mid[i] and close[i-1] <= bb_mid[i-1] if i > 0 else False
        bb_mid_cross_down = close[i] < bb_mid[i] and close[i-1] >= bb_mid[i-1] if i > 0 else False
        
        # RSI extremes
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # 1h HMA trend
        hma_1h_bullish = close[i] > hma_1h[i]
        hma_1h_bearish = close[i] < hma_1h[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: CRSI oversold + 4h bullish + BB low (mean reversion in uptrend)
        if crsi_oversold and htf_bullish and bb_low:
            new_signal = SIZE_ENTRY
        
        # Path 2: Vol spike + RSI oversold + 4h not bearish (panic buy)
        elif vol_spike and rsi_oversold and not htf_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 3: Ranging market + CRSI oversold + BB low (pure mean reversion)
        elif is_ranging and crsi_oversold and bb_low:
            new_signal = SIZE_ENTRY
        
        # Path 4: CRSI crossing up from oversold + 4h bullish + vol normalizing
        elif i > 0 and crsi[i] > crsi[i-1] and crsi[i-1] < 20 and htf_bullish and vol_normal:
            new_signal = SIZE_ENTRY
        
        # Path 5: BB mid cross up + 4h bullish + HMA bullish (momentum confirmation)
        elif bb_mid_cross_up and htf_bullish and hma_1h_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: CRSI overbought + 4h bearish + BB high (mean reversion in downtrend)
        if crsi_overbought and htf_bearish and bb_high:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Vol spike + RSI overbought + 4h not bullish (panic sell)
        elif vol_spike and rsi_overbought and not htf_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Ranging market + CRSI overbought + BB high (pure mean reversion)
        elif is_ranging and crsi_overbought and bb_high:
            new_signal = -SIZE_ENTRY
        
        # Path 4: CRSI crossing down from overbought + 4h bearish + vol normalizing
        elif i > 0 and crsi[i] < crsi[i-1] and crsi[i-1] > 80 and htf_bearish and vol_normal:
            new_signal = -SIZE_ENTRY
        
        # Path 5: BB mid cross down + 4h bearish + HMA bearish (momentum confirmation)
        elif bb_mid_cross_down and htf_bearish and hma_1h_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 1h timeframe)
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
            
            # Calculate trailing stop (2*ATR for 1h timeframe)
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