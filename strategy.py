#!/usr/bin/env python3
"""
Experiment #242: 30m Connors RSI Mean Reversion with 4h/1d HMA Trend Filter
Hypothesis: Connors RSI (CRSI) captures short-term oversold/overbought conditions better than 
standard RSI. Combined with 4h HMA for intermediate trend and 1d HMA for macro bias, this 
should generate more trades than pure trend-following while maintaining positive Sharpe.
CRSI entries: Long when CRSI<15 in uptrend, Short when CRSI>85 in downtrend.
ADX filter (>20) ensures we trade in trending conditions, not chop.
Position sizing: 0.25 entry, 0.125 half at 2R profit. Stoploss: 2.2*ATR trailing.
Target: Beat Sharpe=0.499 with higher trade count and lower drawdown.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h_1d_hma_adx_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    # RSI(3) - short-term RSI
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of Streak - measure consecutive up/down days
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    # RSI on streak (higher streak = more overbought/oversold)
    streak_rsi = calculate_rsi(streak_abs + 1, streak_period)  # +1 to avoid zero
    
    # Percent Rank - where current price change ranks in last N periods
    pct_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        changes = np.diff(close[i-rank_period:i+1])
        current_change = changes[-1] if len(changes) > 0 else 0
        rank = np.sum(changes[:-1] < current_change) / max(1, len(changes) - 1)
        pct_rank[i] = rank * 100
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + pct_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smoothed DM
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / atr
    minus_di = 100 * minus_dm_s / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    adx = calculate_adx(high, low, close, 14)
    
    # Track previous values for crossover detection
    prev_crsi = np.roll(crsi, 1)
    prev_crsi[0] = crsi[0]
    
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
    
    for i in range(150, n):
        # HTF trend filters
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # ADX trend strength filter (>20 = trending, not chop)
        trending = adx[i] > 20
        
        # CRSI signals (mean reversion in trend direction)
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        crsi_rising = crsi[i] > prev_crsi[i]
        crsi_falling = crsi[i] < prev_crsi[i]
        
        # CRSI crossover signals for earlier entry
        crsi_cross_up = prev_crsi[i] < 20 and crsi[i] >= 20
        crsi_cross_down = prev_crsi[i] > 80 and crsi[i] <= 80
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # CRSI oversold in uptrend (mean reversion long)
        if crsi_oversold and trending:
            if trend_4h_bullish and trend_1d_bullish:
                # Strong uptrend - aggressive entry
                new_signal = SIZE_ENTRY
            elif trend_4h_bullish:
                # 4h trend only - moderate entry
                new_signal = SIZE_ENTRY * 0.8
        
        # CRSI cross up from oversold
        elif crsi_cross_up and trending:
            if trend_4h_bullish:
                new_signal = SIZE_ENTRY
        
        # CRSI rising from deep oversold in uptrend
        elif crsi[i] < 30 and crsi_rising and trend_4h_bullish:
            if trend_1d_bullish or adx[i] > 25:
                new_signal = SIZE_ENTRY * 0.8
        
        # === SHORT ENTRY ===
        # CRSI overbought in downtrend (mean reversion short)
        if crsi_overbought and trending:
            if trend_4h_bearish and trend_1d_bearish:
                # Strong downtrend - aggressive entry
                new_signal = -SIZE_ENTRY
            elif trend_4h_bearish:
                # 4h trend only - moderate entry
                new_signal = -SIZE_ENTRY * 0.8
        
        # CRSI cross down from overbought
        elif crsi_cross_down and trending:
            if trend_4h_bearish:
                new_signal = -SIZE_ENTRY
        
        # CRSI falling from deep overbought in downtrend
        elif crsi[i] > 70 and crsi_falling and trend_4h_bearish:
            if trend_1d_bearish or adx[i] > 25:
                new_signal = -SIZE_ENTRY * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.2*ATR from highest)
            current_stop = highest_close - 2.2 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.2 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.2*ATR from lowest)
            current_stop = lowest_close + 2.2 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.2 * atr[i]
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
            trailing_stop = close[i] - 2.2 * atr[i] if position_side > 0 else close[i] + 2.2 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.2 * atr[i] if position_side > 0 else close[i] + 2.2 * atr[i]
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