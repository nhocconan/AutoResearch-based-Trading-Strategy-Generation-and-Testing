#!/usr/bin/env python3
"""
Experiment #276: 1d Daily Regime-Adaptive Strategy with Weekly HMA Filter
Hypothesis: Daily timeframe benefits from regime detection (choppy vs trending).
Use Choppiness Index to switch between mean-reversion (CRSI extremes in chop)
and trend-following (Donchian breakouts in trend). Weekly HMA provides macro bias.
Looser entry thresholds to ensure sufficient trades on daily data.
Position sizing: 0.25 entry, 0.125 half at 2R profit. Stoploss: 2.5*ATR trailing.
Target: Beat Sharpe=0.499 from current best with fewer but higher quality trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_weekly_hma_crsi_donchian_atr_v1"
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
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Mean reversion indicator with 75%+ win rate at extremes.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
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
        streak_window = streak[i-streak_period+1:i+1]
        gains = np.sum(streak_window[streak_window > 0])
        losses = np.abs(np.sum(streak_window[streak_window < 0]))
        if losses == 0:
            streak_rsi[i] = 100.0
        else:
            rs = gains / losses
            streak_rsi[i] = 100 - 100 / (1 + rs)
    
    # Percent Rank (100)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1) * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures if market is trending or ranging.
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    hl_range = highest_high - lowest_low
    
    chop = np.zeros(len(close))
    mask = (hl_range > 0) & (atr_sum > 0)
    chop[mask] = 100 * np.log10(atr_sum[mask] / hl_range[mask]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    # Fill initial values
    for i in range(period):
        upper[i] = np.max(high[:i+1])
        lower[i] = np.min(low[:i+1])
        mid[i] = (upper[i] + lower[i]) / 2
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma50[:50] = close[:50]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    ATR_MULT = 2.5
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_price = 0.0
    lowest_price = 0.0
    initial_risk = 0.0
    
    for i in range(150, n):
        # Weekly trend bias
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Regime detection
        is_choppy = chop[i] > 50  # Looser threshold for more regime switches
        is_trending = chop[i] < 50
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        
        # Donchian breakout signals
        breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # Trend pullback entries
        pullback_long = close[i] < donch_mid[i] and close[i] > donch_lower[i]
        pullback_short = close[i] > donch_mid[i] and close[i] < donch_upper[i]
        
        # SMA50 trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === MEAN REVERSION (Choppy Market) ===
        if is_choppy:
            # Long on CRSI oversold + weekly bullish bias
            if crsi_oversold:
                if weekly_bullish or above_sma50:
                    new_signal = SIZE_ENTRY
                elif crsi[i] < 15:  # Extreme oversold
                    new_signal = SIZE_ENTRY
            
            # Short on CRSI overbought + weekly bearish bias
            if crsi_overbought:
                if weekly_bearish or below_sma50:
                    new_signal = -SIZE_ENTRY
                elif crsi[i] > 85:  # Extreme overbought
                    new_signal = -SIZE_ENTRY
        
        # === TREND FOLLOWING (Trending Market) ===
        elif is_trending:
            # Donchian breakout long with weekly confirmation
            if breakout_long:
                if weekly_bullish:
                    new_signal = SIZE_ENTRY
                elif above_sma50:
                    new_signal = SIZE_ENTRY * 0.6  # Weaker signal without weekly confirmation
            
            # Donchian breakout short with weekly confirmation
            if breakout_short:
                if weekly_bearish:
                    new_signal = -SIZE_ENTRY
                elif below_sma50:
                    new_signal = -SIZE_ENTRY * 0.6
            
            # Trend pullback entries (buy dips in uptrend, sell rallies in downtrend)
            if pullback_long and weekly_bullish and crsi[i] < 50:
                new_signal = SIZE_ENTRY * 0.8
            
            if pullback_short and weekly_bearish and crsi[i] > 50:
                new_signal = -SIZE_ENTRY * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest price for trailing
            if high[i] > highest_price:
                highest_price = high[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_price - ATR_MULT * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if low[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced and initial_risk > 0:
                # Take profit at 2R
                profit = close[i] - entry_price
                if profit >= 2.0 * initial_risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price for trailing
            if lowest_price == 0.0 or low[i] < lowest_price:
                lowest_price = low[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_price + ATR_MULT * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if high[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced and initial_risk > 0:
                # Take profit at 2R
                profit = entry_price - close[i]
                if profit >= 2.0 * initial_risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            initial_risk = ATR_MULT * atr[i]
            trailing_stop = close[i] - initial_risk if position_side > 0 else close[i] + initial_risk
            highest_price = high[i] if position_side > 0 else 0.0
            lowest_price = low[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            initial_risk = ATR_MULT * atr[i]
            trailing_stop = close[i] - initial_risk if position_side > 0 else close[i] + initial_risk
            highest_price = high[i] if position_side > 0 else 0.0
            lowest_price = low[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            position_reduced = False
            initial_risk = 0.0
        
        signals[i] = new_signal
    
    return signals