#!/usr/bin/env python3
"""
Experiment #037: 15m Connors RSI Mean Reversion with 4h HMA Regime Filter
Hypothesis: 15m timeframe captures short-term mean reversion opportunities while 4h HMA provides trend bias.
Key insight: Connors RSI (CRSI) has 75% win rate for mean reversion. Combined with Choppiness Index regime filter,
we can switch between mean reversion (CHOP>61.8) and trend following (CHOP<38.2).
Position sizing: 0.25-0.30 discrete levels with 2.5*ATR stoploss.
Timeframe: 15m (REQUIRED for exp#037), HTF: 4h via mtf_data helper.
Why this might work: 15m has enough trades (>10/year), CRSI catches reversals, CHOP filter avoids whipsaws.
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs failed experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_chop_regime_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    """Calculate Streak RSI for Connors RSI."""
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    streak_rsi = np.zeros(n)
    
    for i in range(period, n):
        if i >= period:
            window = abs_streak[i-period+1:i+1]
            if len(window) > 0 and np.max(window) > 0:
                streak_rsi[i] = 100 * np.sum(window > 0) / len(window)
            else:
                streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank for Connors RSI."""
    n = len(close)
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        pct_rank[i] = rank * 100
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index for regime detection."""
    atr = calculate_atr(high, low, close, period)
    
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    chop = np.zeros(len(close))
    chop[:] = np.nan
    
    mask = (price_range > 0) & (atr_sum > 0)
    chop[mask] = 100 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    
    return chop

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # Choppiness regime detection
        choppy_regime = chop[i] > 55.0  # Range-bound market
        trending_regime = chop[i] < 45.0  # Trending market
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # CRSI conditions - LOOSENED for more trades
        crsi_oversold = crsi[i] < 25.0  # Long entry (was <10, too strict)
        crsi_overbought = crsi[i] > 75.0  # Short entry (was >90, too strict)
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # Bollinger Band position
        near_bb_lower = close[i] <= bb_lower[i] * 1.005
        near_bb_upper = close[i] >= bb_upper[i] * 0.995
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        
        # Volume confirmation (optional filter)
        vol_confirm = True  # Simplified - can add volume filter later
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: CRSI oversold + 4h bullish + near BB lower
        if crsi_oversold and bull_trend_4h and near_bb_lower:
            new_signal = SIZE_BASE
        
        # Secondary: CRSI extreme oversold in any regime (catch crashes)
        elif crsi_extreme_oversold and above_200:
            new_signal = SIZE_BASE
        
        # Tertiary: Mean reversion in choppy regime
        elif choppy_regime and crsi[i] < 30.0 and near_bb_lower:
            new_signal = SIZE_HALF
        
        # Quaternary: Pullback to EMA21 in uptrend
        elif bull_trend_4h and price_near_ema21_long and ema_bullish and crsi[i] < 40.0:
            new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        # Primary: CRSI overbought + 4h bearish + near BB upper
        if new_signal == 0.0:  # Don't override long signal
            if crsi_overbought and bear_trend_4h and near_bb_upper:
                new_signal = -SIZE_BASE
            
            # Secondary: CRSI extreme overbought in any regime
            elif crsi_extreme_overbought and below_200:
                new_signal = -SIZE_BASE
            
            # Tertiary: Mean reversion in choppy regime
            elif choppy_regime and crsi[i] > 70.0 and near_bb_upper:
                new_signal = -SIZE_HALF
            
            # Quaternary: Bounce to EMA21 in downtrend
            elif bear_trend_4h and price_near_ema21_short and ema_bearish and crsi[i] > 60.0:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals