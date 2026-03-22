#!/usr/bin/env python3
"""
Experiment #494: 30m Donchian Breakout + 4h HMA Trend + Choppiness Regime Filter + ATR Stop
Hypothesis: Donchian breakouts work well on daily (exp #486 Sharpe=0.198). Adapting to 30m
with 4h HMA trend bias and Choppiness Index regime filter should reduce false breakouts.
Choppiness > 61.8 = range (avoid breakouts), Choppiness < 38.2 = trend (allow breakouts).
4h HMA provides HTF trend alignment. ATR stops control downside. Conservative sizing (0.25).
Multiple entry paths ensure >=10 trades per symbol. Must beat Sharpe=0.499 baseline.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_4h_hma_chop_regime_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout indicator.
    Returns: upper channel, lower channel, middle
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

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
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # 30m HMA for additional trend confirmation
    hma_30m = calculate_hma(close, 21)
    hma_30m_fast = calculate_hma(close, 10)
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(hma_30m[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m HMA trend
    hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        hma_rising = hma_30m[i] > hma_30m[i - 1] if i > 0 else False
        hma_falling = hma_30m[i] < hma_30m[i - 1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_30m_fast[i] > hma_30m[i]
        fast_below_slow = hma_30m_fast[i] < hma_30m[i]
        
        # Choppiness regime
        chop_trending = chop[i] < 45.0  # Allow some flexibility vs strict 38.2
        chop_ranging = chop[i] > 55.0
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # Donchian position (price relative to channel)
        in_upper_half = close[i] > donchian_mid[i]
        in_lower_half = close[i] < donchian_mid[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # SMA 200 filter
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # Volume confirmation (above average)
        vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_above_avg = volume[i] > vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + HTF bullish + Trending regime + RSI bullish
        if breakout_long and htf_bullish and chop_trending and rsi_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: HTF bullish + 30m HMA bullish + Fast HMA crossover + Volume
        elif htf_bullish and hma_30m_bullish and fast_above_slow and vol_above_avg:
            new_signal = SIZE_ENTRY
        
        # Path 3: HTF bullish + Price above SMA200 + RSI pullback (40-50)
        elif htf_bullish and above_sma200 and rsi[i] > 40 and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        
        # Path 4: Donchian upper half + HTF bullish + HMA rising + Chop not extreme
        elif in_upper_half and htf_bullish and hma_rising and chop[i] < 60:
            new_signal = SIZE_ENTRY
        
        # Path 5: Breakout long + Volume surge + HTF not bearish
        elif breakout_long and volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False and not htf_bearish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + HTF bearish + Trending regime + RSI bearish
        if breakout_short and htf_bearish and chop_trending and rsi_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: HTF bearish + 30m HMA bearish + Fast HMA crossover down + Volume
        elif htf_bearish and hma_30m_bearish and fast_below_slow and vol_above_avg:
            new_signal = -SIZE_ENTRY
        
        # Path 3: HTF bearish + Price below SMA200 + RSI pullback (50-60)
        elif htf_bearish and below_sma200 and rsi[i] > 50 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Donchian lower half + HTF bearish + HMA falling + Chop not extreme
        elif in_lower_half and htf_bearish and hma_falling and chop[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Breakout short + Volume surge + HTF not bullish
        elif breakout_short and volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False and not htf_bullish:
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