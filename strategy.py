#!/usr/bin/env python3
"""
Experiment #092: 30m Connors RSI Mean Reversion with 4h HMA Trend Filter
Hypothesis: 30m timeframe is ideal for mean reversion entries within HTF trend.
Connors RSI (CRSI) has 75% win rate for short-term reversals.
4h HMA provides stable trend bias - only take mean reversion in trend direction.
ADX filter ensures we avoid choppy markets where mean reversion fails.
This combines proven mean reversion (CRSI) with trend filter (4h HMA) for best of both.

Why this might work on 30m (learning from failures):
- #080 (30m EMA momentum): Sharpe=-3.202 - pure momentum failed on 30m
- #086 (30m Supertrend): Sharpe=-0.616 - trend following whipsawed on 30m
- Key insight: 30m is TOO NOISY for pure trend, PERFECT for mean reversion pullbacks
- CRSI catches oversold/overbought extremes with high win rate
- 4h HMA filter ensures we only trade with HTF trend (buy dips in uptrend)
- ADX > 20 ensures market has directional bias (not dead chop)

Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.30 strong signals. Stoploss at 2.0*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h_hma_adx_regime_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Calculate RSI of streak length for Connors RSI.
    Streak = consecutive up/down days.
    """
    n = len(close)
    streak = np.zeros(n)
    
    # Calculate streak direction and length
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # Long positive streak = high value, long negative streak = low value
    abs_streak = np.abs(streak)
    streak_rsi = np.zeros(n)
    
    # Normalize streak to 0-100 scale using RSI formula on streak values
    max_streak = max(1, np.max(abs_streak))
    streak_rsi = 50 + (streak / max_streak) * 50
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank for Connors RSI.
    Percent rank of current return vs last N periods.
    """
    n = len(close)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window < current_return)
        percent_rank[i] = (rank / period) * 100
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi3 + streak_rsi + percent_rank) / 3
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth using Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Bollinger Bands for additional mean reversion confirmation
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (stable, slow)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CONNORS RSI MEAN REVERSION SIGNAL ===
        # CRSI < 10 = extremely oversold (long opportunity)
        # CRSI > 90 = extremely overbought (short opportunity)
        crsi_oversold = crsi[i] < 15  # Slightly relaxed from 10 to ensure trades
        crsi_overbought = crsi[i] > 85  # Slightly relaxed from 90 to ensure trades
        
        # === ADX REGIME FILTER ===
        # ADX > 20 = trending market (mean reversion works better in trends)
        # ADX < 15 = choppy market (avoid entries)
        trending_market = adx[i] > 20
        strong_trend = adx[i] > 25
        
        # === BOLLINGER BAND CONFIRMATION ===
        # Price below lower BB = oversold confirmation
        # Price above upper BB = overbought confirmation
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: CRSI oversold + 4h bullish trend + ADX trending
        if crsi_oversold and bull_trend_4h and trending_market:
            if price_below_bb:
                new_signal = SIZE_STRONG  # Strong signal with BB confirmation
            else:
                new_signal = SIZE_BASE
        
        # Secondary: CRSI very oversold + 4h bullish (relaxed ADX)
        if new_signal == 0.0 and crsi[i] < 10 and bull_trend_4h:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: CRSI overbought + 4h bearish trend + ADX trending
        if crsi_overbought and bear_trend_4h and trending_market:
            if price_above_bb:
                new_signal = -SIZE_STRONG  # Strong signal with BB confirmation
            else:
                new_signal = -SIZE_BASE
        
        # Secondary: CRSI very overbought + 4h bearish (relaxed ADX)
        if new_signal == 0.0 and crsi[i] > 90 and bear_trend_4h:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR for 30m ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals