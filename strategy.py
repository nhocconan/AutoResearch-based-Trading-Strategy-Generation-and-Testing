#!/usr/bin/env python3
"""
Experiment #181: 15m Multi-Timeframe Trend-Pullback Strategy
Hypothesis: 15m captures intraday swings while 4h/1h HTF filters prevent counter-trend trades.
Uses 4h HMA for major trend, 1h RSI for momentum confirmation, 15m entries on pullbacks.
Connors RSI (CRSI) for mean-reversion entries in ranging markets. Choppiness Index switches
between trend-following (CHOP<40) and mean-reversion (CHOP>60) modes. ATR stoploss at 3*ATR.
Position sizing: 0.25 entry, reduced to 0.125 at 2R profit. Discrete levels minimize fee churn.
This targets the 2022 crash (trend mode with HTF filter) and 2025 consolidation (range mode).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_chop_4h_1h_hma_trend_v1"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long entry: CRSI < 10 (oversold)
    Short entry: CRSI > 90 (overbought)
    Reference: Connors & Alvarez (2012)
    """
    n = len(close)
    
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
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, streak_abs, 0.0)
    avg_sg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_sl = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    rs_streak = np.where(avg_sl > 0, avg_sg / avg_sl, 100.0)
    rsi_streak = 100 - 100 / (1 + rs_streak)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100 * count_below / (rank_period - 1)
    
    # CRSI
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    crsi[:rank_period] = 50.0  # Fill warmup
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Reference: E.W. Dreiss
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    chop = 100 * np.log10(atr_sum / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    Reference: John Ehlers (2002)
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_hl = highest - lowest
        range_hl = max(range_hl, 1e-10)
        
        price = (close[i] - lowest) / range_hl
        price = 0.999 * price + 0.001  # Bound to (0, 1)
        
        value = 0.5 * np.log((1 + price) / (1 - price + 1e-10))
        
        fisher[i] = 0.67 * value + 0.33 * fisher[i-1] if i > period else value
        trigger[i] = 0.5 * fisher[i] + 0.5 * fisher[i-1] if i > period else fisher[i]
    
    return fisher, trigger

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
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
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(250, n):  # Warmup for all indicators
        # HTF trend filters (4h major trend, 1h momentum)
        hma_4h_valid = hma_4h_aligned[i] > 0
        hma_1h_valid = hma_1h_aligned[i] > 0
        
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        trend_1h_bullish = hma_1h_valid and close[i] > hma_1h_aligned[i]
        trend_1h_bearish = hma_1h_valid and close[i] < hma_1h_aligned[i]
        
        rsi_1h_bullish = rsi_1h_aligned[i] > 50
        rsi_1h_bearish = rsi_1h_aligned[i] < 50
        
        # Regime detection
        is_ranging = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # 15m trend
        trend_15m_bullish = hma_20[i] > hma_50[i] and hma_20[i] > 0
        trend_15m_bearish = hma_20[i] < hma_50[i] and hma_50[i] > 0
        
        # Price vs SMA200
        above_sma200 = sma_200[i] > 0 and close[i] > sma_200[i]
        below_sma200 = sma_200[i] > 0 and close[i] < sma_200[i]
        
        # CRSI signals (mean reversion)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_rising = crsi[i] > crsi[i-3] if i > 3 else False
        crsi_falling = crsi[i] < crsi[i-3] if i > 3 else False
        
        # Fisher Transform signals
        fisher_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # MACD signals
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # RSI signals
        rsi_oversold = rsi_15m[i] < 35
        rsi_overbought = rsi_15m[i] > 65
        rsi_rising = rsi_15m[i] > rsi_15m[i-3] if i > 3 else False
        rsi_falling = rsi_15m[i] < rsi_15m[i-3] if i > 3 else False
        
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (ranging market) ===
        if is_ranging:
            # Long: CRSI oversold + price > SMA200 + 4h not bearish
            if crsi_oversold and above_sma200:
                if not trend_4h_bearish and (crsi_rising or rsi_rising):
                    new_signal = SIZE_ENTRY
            
            # Short: CRSI overbought + price < SMA200 + 4h not bullish
            elif crsi_overbought and below_sma200:
                if not trend_4h_bullish and (crsi_falling or rsi_falling):
                    new_signal = -SIZE_ENTRY
            
            # Fisher reversals in range
            elif fisher_long and not trend_4h_bearish:
                if above_sma200 or rsi_oversold:
                    new_signal = SIZE_ENTRY
            elif fisher_short and not trend_4h_bullish:
                if below_sma200 or rsi_overbought:
                    new_signal = -SIZE_ENTRY
        
        # === TREND FOLLOWING MODE (trending market) ===
        elif is_trending:
            # Long: 4h bullish + 1h bullish + 15m pullback
            if trend_4h_bullish and trend_1h_bullish:
                if rsi_15m[i] < 50 and rsi_rising:
                    new_signal = SIZE_ENTRY
                elif macd_bullish and trend_15m_bullish:
                    new_signal = SIZE_ENTRY
            
            # Short: 4h bearish + 1h bearish + 15m pullback
            elif trend_4h_bearish and trend_1h_bearish:
                if rsi_15m[i] > 50 and rsi_falling:
                    new_signal = -SIZE_ENTRY
                elif macd_bearish and trend_15m_bearish:
                    new_signal = -SIZE_ENTRY
            
            # HMA crossover with HTF confirmation
            if trend_4h_bullish and hma_20[i] > hma_50[i] and hma_20[i-1] <= hma_50[i-1]:
                new_signal = SIZE_ENTRY
            elif trend_4h_bearish and hma_20[i] < hma_50[i] and hma_20[i-1] >= hma_50[i-1]:
                new_signal = -SIZE_ENTRY
        
        # === HYBRID MODE (transition) ===
        if new_signal == 0.0 and 45.0 <= chop[i] <= 55.0:
            # Use both trend and MR signals with weaker filters
            if trend_4h_bullish and (rsi_oversold or crsi_oversold):
                new_signal = SIZE_ENTRY
            elif trend_4h_bearish and (rsi_overbought or crsi_overbought):
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest price for trailing
            if high[i] > highest_price:
                highest_price = high[i]
            
            # Calculate trailing stop (3*ATR from highest)
            current_stop = highest_price - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if low[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price for trailing
            if lowest_price == 0.0 or low[i] < lowest_price:
                lowest_price = low[i]
            
            # Calculate trailing stop (3*ATR from lowest)
            current_stop = lowest_price + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if high[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
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
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_price = high[i] if position_side > 0 else 0.0
            lowest_price = low[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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
        
        signals[i] = new_signal
    
    return signals