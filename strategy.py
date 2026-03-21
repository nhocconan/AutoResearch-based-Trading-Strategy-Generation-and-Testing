#!/usr/bin/env python3
"""
Experiment #321: 1h Fisher Transform + 4h HMA Trend + Choppiness Regime Filter + ATR Stops
Hypothesis: 1h Fisher Transform catches reversals better than RSI in bear/range markets (2025 test).
4h HMA provides macro trend bias (proven in best strategies). Choppiness Index detects regime:
CHOP>61.8 = range (use mean reversion), CHOP<38.2 = trend (use breakout). This adapts to market state.
1h timeframe = more trades than 4h/12h/1d, but need regime filter to avoid whipsaws.
Target: Beat Sharpe=0.499 with better reversal timing and regime adaptation.
Timeframe: 1h (required for this experiment), HTF: 4h for trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_chop_regime_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear/range markets.
    Transform price to Gaussian distribution, crossings of ±1.5 signal reversals.
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Normalize price within period range
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = 0.66 * ((hl2 - lowest) / range_val) + 0.67
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher = np.log((1 + normalized) / (1 - normalized))
    fisher_s = pd.Series(fisher)
    fisher_smooth = fisher_s.ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher_smooth

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - detects ranging vs trending markets.
    CHOP > 61.8 = choppy/range (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            chop[i] = 100
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        atr_avg = atr_sum / period
        chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop[:period] = 50  # Default value for warmup
    return chop

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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_fisher = np.roll(fisher, 1)
    prev_fisher[0] = fisher[0]
    
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
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        trend_bullish = not np.isnan(hma_4h_aligned[i]) and close[i] > hma_4h_aligned[i]
        trend_bearish = not np.isnan(hma_4h_aligned[i]) and close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        is_choppy = chop[i] > 55  # Range market (mean reversion)
        is_trending = chop[i] < 45  # Trend market (breakout)
        
        # Fisher Transform reversal signals
        fisher_cross_up = fisher[i] > -1.5 and prev_fisher[i] <= -1.5  # Oversold reversal
        fisher_cross_down = fisher[i] < 1.5 and prev_fisher[i] >= 1.5  # Overbought reversal
        fisher_extreme_low = fisher[i] < -2.0  # Deep oversold
        fisher_extreme_high = fisher[i] > 2.0  # Deep overbought
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 < rsi[i] < 65
        
        # Price momentum
        price_up = close[i] > prev_close[i]
        price_down = close[i] < prev_close[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Regime 1: Trending market + 4h bullish + Fisher reversal from oversold
        if is_trending and trend_bullish and fisher_cross_up:
            new_signal = SIZE_ENTRY
        # Regime 2: Choppy market + Fisher extreme low + RSI oversold (mean reversion)
        elif is_choppy and fisher_extreme_low and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Regime 3: 4h bullish + Fisher cross up + price momentum (simple trend entry)
        elif trend_bullish and fisher_cross_up and price_up:
            new_signal = SIZE_ENTRY
        # Regime 4: Fisher cross up + RSI rising from oversold (reversal confirmation)
        elif fisher_cross_up and rsi[i] > 30 and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Regime 5: 4h bullish + RSI neutral + price up (momentum continuation)
        elif trend_bullish and rsi_neutral and price_up and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Regime 1: Trending market + 4h bearish + Fisher reversal from overbought
        if is_trending and trend_bearish and fisher_cross_down:
            new_signal = -SIZE_ENTRY
        # Regime 2: Choppy market + Fisher extreme high + RSI overbought (mean reversion)
        elif is_choppy and fisher_extreme_high and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Regime 3: 4h bearish + Fisher cross down + price momentum (simple trend entry)
        elif trend_bearish and fisher_cross_down and price_down:
            new_signal = -SIZE_ENTRY
        # Regime 4: Fisher cross down + RSI falling from overbought (reversal confirmation)
        elif fisher_cross_down and rsi[i] < 70 and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Regime 5: 4h bearish + RSI neutral + price down (momentum continuation)
        elif trend_bearish and rsi_neutral and price_down and rsi[i] < 55:
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