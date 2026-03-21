#!/usr/bin/env python3
"""
Experiment #043: 15m Connors RSI + 4h HMA Trend + Choppiness Filter
Hypothesis: 15m timeframe failed before due to excessive churn and weak filters.
This strategy uses Connors RSI (proven 75% win rate mean reversion) with strict
4h HMA trend filter to only trade with macro direction. Choppiness Index > 61.8
filters out choppy markets where mean reversion fails. Bollinger squeeze detection
captures breakout momentum. Very selective entries (need 3+ conditions) to reduce
trade count and fee drag. Position size 0.22 with 2.5x ATR stoploss.
Key difference from failed #037: Connors RSI instead of simple RSI, CHOP filter,
fewer entry triggers, stronger HTF alignment requirement.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_chop_bb_v1"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate.
    """
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        streak_vals = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_vals > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100 if streak_period > 0 else 50
    
    # Percent Rank: where does current close rank in last 100 bars?
    pct_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        pct_rank[i] = (rank / rank_period) * 100
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures if market is trending or choppy.
    CHOP > 61.8 = choppy/range (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    choppiness = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50
    
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = np.where(sma > 0, (upper - lower) / sma, 0)
    bb_pct = np.where((upper - lower) > 0, (close - lower) / (upper - lower), 0.5)
    return upper, lower, sma, bw, bb_pct

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h to 15m (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_width, bb_pct = calculate_bollinger_bands(close, 20, 2.0)
    sma_200 = calculate_sma(close, 200)
    
    # 15m HMA for short-term trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # BB Width percentile for squeeze detection
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).rank(pct=True).values
    bb_width_pct = np.nan_to_num(bb_width_pct, nan=0.5)
    
    signals = np.zeros(n)
    SIZE = 0.22
    HALF_SIZE = 0.11
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    max_profit = 0.0
    
    for i in range(200, n):
        # 4h macro trend filter (MUST align with HTF)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 15m trend alignment
        trend_15m_bullish = hma_21[i] > hma_50[i] and close[i] > sma_200[i] if sma_200[i] > 0 else False
        trend_15m_bearish = hma_21[i] < hma_50[i] and close[i] < sma_200[i] if sma_200[i] > 0 else False
        
        # Choppiness regime filter
        choppy_market = choppiness[i] > 55  # Range market (mean reversion works)
        trending_market = choppiness[i] < 45  # Trend market (trend following works)
        
        # Connors RSI extremes (mean reversion signals)
        crsi_oversold = crsi[i] < 15  # Very oversold
        crsi_overbought = crsi[i] > 85  # Very overbought
        crsi_extreme = crsi_oversold or crsi_overbought
        
        # Bollinger Band signals
        bb_squeeze = bb_width_pct[i] < 0.25  # Very tight squeeze = breakout coming
        bb_lower_touch = bb_pct[i] < 0.15  # Price at lower band
        bb_upper_touch = bb_pct[i] > 0.85  # Price at upper band
        bb_breakout_long = close[i] > bb_upper[i] and bb_width[i] > bb_width[i-5] if i > 5 else False
        bb_breakout_short = close[i] < bb_lower[i] and bb_width[i] > bb_width[i-5] if i > 5 else False
        
        # Price momentum
        mom_5 = (close[i] - close[i-5]) / close[i-5] * 100 if i > 5 and close[i-5] > 0 else 0
        mom_strong_long = mom_5 > 2.0
        mom_strong_short = mom_5 < -2.0
        
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (need 3+ conditions to reduce churn)
        # Trigger 1: CRSI oversold + 4h bullish + choppy market (mean reversion in uptrend)
        if crsi_oversold and trend_4h_bullish and choppy_market:
            new_signal = SIZE
        # Trigger 2: BB lower touch + 4h bullish + trend_15m bullish (pullback in uptrend)
        elif bb_lower_touch and trend_4h_bullish and trend_15m_bullish:
            new_signal = SIZE
        # Trigger 3: BB squeeze breakout long + 4h bullish (momentum breakout)
        elif bb_breakout_long and trend_4h_bullish:
            new_signal = SIZE
        # Trigger 4: CRSI oversold + BB lower touch + 4h bullish (strong mean reversion)
        elif crsi_oversold and bb_lower_touch and trend_4h_bullish:
            new_signal = SIZE
        # Trigger 5: HMA crossover long + 4h bullish + trending market (trend follow)
        elif hma_21[i] > hma_50[i] and hma_21[i-1] <= hma_50[i-1] and trend_4h_bullish and trending_market:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: CRSI overbought + 4h bearish + choppy market (mean reversion in downtrend)
        if crsi_overbought and trend_4h_bearish and choppy_market:
            new_signal = -SIZE
        # Trigger 2: BB upper touch + 4h bearish + trend_15m bearish (rally in downtrend)
        elif bb_upper_touch and trend_4h_bearish and trend_15m_bearish:
            new_signal = -SIZE
        # Trigger 3: BB squeeze breakout short + 4h bearish (momentum breakdown)
        elif bb_breakout_short and trend_4h_bearish:
            new_signal = -SIZE
        # Trigger 4: CRSI overbought + BB upper touch + 4h bearish (strong mean reversion)
        elif crsi_overbought and bb_upper_touch and trend_4h_bearish:
            new_signal = -SIZE
        # Trigger 5: HMA crossover short + 4h bearish + trending market (trend follow)
        elif hma_21[i] < hma_50[i] and hma_21[i-1] >= hma_50[i-1] and trend_4h_bearish and trending_market:
            new_signal = -SIZE
        
        # Stoploss and take profit logic (Rule 6)
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                if close[i] > entry_price:
                    max_profit = max(max_profit, close[i] - entry_price)
                if max_profit >= 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
                if close[i] < entry_price:
                    max_profit = max(max_profit, entry_price - close[i])
                if max_profit >= 2.5 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            max_profit = 0.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
                max_profit = 0.0
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            max_profit = 0.0
        
        signals[i] = new_signal
    
    return signals