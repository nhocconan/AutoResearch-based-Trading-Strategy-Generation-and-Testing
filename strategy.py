#!/usr/bin/env python3
"""
Experiment #163: 15m Connors RSI Mean Reversion with 4h HMA Trend Filter
Hypothesis: 15m timeframe is too noisy for pure trend following (see exp#151, #157 failures).
Connors RSI (CRSI) is proven for mean reversion with 75%+ win rate on short timeframes.
4h HMA provides major trend bias to avoid counter-trend mean reversion.
Volume confirmation filters false breakouts. ATR stoploss at 2.0*ATR limits drawdown.
Entry conditions loosened (CRSI<15/>85 instead of <10/>90) to ensure sufficient trades.
Position sizing: 0.25 entry, 0.125 half-size at 2R profit. Discrete levels minimize fees.
This targets range markets (2025) while respecting trend direction from 4h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_volume_atr_v1"
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
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Reference: Connors, Alvarez, Radtke - "Short Term Trading Strategies That Work"
    Long entry: CRSI < 10-15
    Short entry: CRSI > 85-90
    """
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    avg_sg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_sl = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_rsi = np.zeros(n)
    mask = avg_sl > 0
    streak_rsi[mask] = 100 - 100 / (1 + avg_sg[mask] / avg_sl[mask])
    streak_rsi[~mask] = 100.0
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component (100-period)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands for mean reversion reference."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    vol_sma = calculate_volume_sma(volume, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
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
        # 4h trend filter
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_bullish_4h = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish_4h = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 15m trend
        trend_bullish_15m = hma_20[i] > hma_50[i]
        trend_bearish_15m = hma_20[i] < hma_50[i]
        
        # CRSI signals (loosened for more trades)
        crsi_oversold = crsi[i] < 18
        crsi_overbought = crsi[i] > 82
        crsi_rising = crsi[i] > crsi[i-2] if i > 2 else False
        crsi_falling = crsi[i] < crsi[i-2] if i > 2 else False
        
        # Volume confirmation
        volume_above_avg = volume[i] > vol_sma[i] * 0.8  # Allow slightly below avg for entries
        
        # Bollinger Band position
        near_lower_bb = close[i] < bb_lower[i] * 1.005
        near_upper_bb = close[i] > bb_upper[i] * 0.995
        below_mid_bb = close[i] < bb_mid[i]
        above_mid_bb = close[i] > bb_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY: Mean reversion in bullish 4h trend ===
        if crsi_oversold and volume_above_avg:
            # Priority 1: Near lower BB + 4h bullish
            if near_lower_bb and trend_bullish_4h:
                new_signal = SIZE_ENTRY
            # Priority 2: 4h bullish + 15m pullback (not oversold on 15m trend)
            elif trend_bullish_4h and below_mid_bb and crsi_rising:
                new_signal = SIZE_ENTRY
            # Priority 3: Pure mean reversion (no strong 4h bearish)
            elif not trend_bearish_4h and near_lower_bb:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY: Mean reversion in bearish 4h trend ===
        elif crsi_overbought and volume_above_avg:
            # Priority 1: Near upper BB + 4h bearish
            if near_upper_bb and trend_bearish_4h:
                new_signal = -SIZE_ENTRY
            # Priority 2: 4h bearish + 15m rally (not overbought on 15m trend)
            elif trend_bearish_4h and above_mid_bb and crsi_falling:
                new_signal = -SIZE_ENTRY
            # Priority 3: Pure mean reversion (no strong 4h bullish)
            elif not trend_bullish_4h and near_upper_bb:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
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
            
            # Calculate trailing stop (2.0*ATR from lowest)
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
        prev_signal = signals[i-1] if i > 0 else 0.0
        
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