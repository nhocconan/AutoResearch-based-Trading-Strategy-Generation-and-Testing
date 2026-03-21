#!/usr/bin/env python3
"""
Experiment #046: 4h Multi-Regime Strategy with Daily/Weekly HMA + Connors RSI + Volume
Hypothesis: 4h timeframe balances swing trading frequency with noise reduction.
Using Connors RSI (CRSI) for mean reversion entries combined with Supertrend trend filter.
Daily and Weekly HMA provide multi-timeframe regime filtering. Volume confirmation
filters false breakouts. Relaxed CRSI thresholds (5/95 instead of 10/90) ensure
sufficient trade generation across all symbols. Position sizing 0.28 with 2.5x ATR
stoploss balances risk/reward while keeping drawdown controlled.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_daily_weekly_crsi_volume_v1"
timeframe = "4h"
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
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Better for mean reversion than standard RSI.
    """
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = max(0, streak[i-1] + 1)
        elif close[i] < close[i-1]:
            streak[i] = min(0, streak[i-1] - 1)
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100 * streak_abs[i] / max(1, streak_period)
        else:
            streak_rsi[i] = 100 * (1 - streak_abs[i] / max(1, streak_period))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        pct_rank[i] = 100 * np.sum(window[:-1] < close[i]) / (rank_period - 1)
    
    # Combine components
    crsi = (rsi_short + streak_rsi + pct_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 4h HMA for trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # 4h SMA for mean reversion reference
    sma_200 = pd.Series(close).rolling(window=200, min_periods=100).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    max_profit = 0.0
    
    for i in range(150, n):
        # Weekly macro regime (bull/bear)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        weekly_neutral = hma_1w_aligned[i] <= 0
        
        # Daily trend filter
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 4h Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # 4h HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # CRSI mean reversion signals (relaxed thresholds for more trades)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_neutral = crsi[i] > 30 and crsi[i] < 70
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.2  # Above average volume
        
        # Price position
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (multiple paths to ensure trades)
        # Trigger 1: Supertrend flip long (strongest signal)
        if st_flip_long:
            new_signal = SIZE
        # Trigger 2: Weekly bullish + CRSI oversold + Supertrend long (pullback in bull)
        elif weekly_bullish and crsi_oversold and st_long:
            new_signal = SIZE
        # Trigger 3: Daily bullish + CRSI oversold + volume confirmed
        elif daily_bullish and crsi_oversold and vol_confirmed:
            new_signal = SIZE
        # Trigger 4: HMA trend long + CRSI neutral + price above HMA21 (trend continuation)
        elif hma_trend_long and crsi_neutral and price_above_hma21:
            new_signal = SIZE
        # Trigger 5: CRSI oversold + price above SMA200 + RSI oversold (deep pullback)
        elif crsi_oversold and price_above_sma200 and rsi_oversold:
            new_signal = SIZE
        # Trigger 6: Weekly neutral + Daily bullish + Supertrend long + CRSI rising
        elif weekly_neutral and daily_bullish and st_long and crsi[i] > crsi[i-5]:
            new_signal = SIZE
        # Trigger 7: Supertrend long + HMA trend long + volume confirmed (momentum)
        elif st_long and hma_trend_long and vol_confirmed:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Supertrend flip short (strongest signal)
        if st_flip_short:
            new_signal = -SIZE
        # Trigger 2: Weekly bearish + CRSI overbought + Supertrend short (rally in bear)
        elif weekly_bearish and crsi_overbought and st_short:
            new_signal = -SIZE
        # Trigger 3: Daily bearish + CRSI overbought + volume confirmed
        elif daily_bearish and crsi_overbought and vol_confirmed:
            new_signal = -SIZE
        # Trigger 4: HMA trend short + CRSI neutral + price below HMA21 (trend continuation)
        elif hma_trend_short and crsi_neutral and price_below_hma21:
            new_signal = -SIZE
        # Trigger 5: CRSI overbought + price below SMA200 + RSI overbought (rally into resistance)
        elif crsi_overbought and price_below_sma200 and rsi_overbought:
            new_signal = -SIZE
        # Trigger 6: Weekly neutral + Daily bearish + Supertrend short + CRSI falling
        elif weekly_neutral and daily_bearish and st_short and crsi[i] < crsi[i-5]:
            new_signal = -SIZE
        # Trigger 7: Supertrend short + HMA trend short + volume confirmed (momentum)
        elif st_short and hma_trend_short and vol_confirmed:
            new_signal = -SIZE
        
        # Stoploss and take profit logic (Rule 6)
        if position_side > 0 and entry_price > 0:
            # Initial stoploss
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                # Track max profit for take profit
                if close[i] > entry_price:
                    max_profit = max(max_profit, close[i] - entry_price)
                # Take partial profit at 2.5R
                if max_profit >= 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            # Initial stoploss
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
                # Track max profit for take profit
                if close[i] < entry_price:
                    max_profit = max(max_profit, entry_price - close[i])
                # Take partial profit at 2.5R
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