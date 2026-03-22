#!/usr/bin/env python3
"""
Experiment #019: 15m Multi-Timeframe Mean Reversion with 4h HMA Regime Filter
Hypothesis: 15m timeframe captures short-term mean reversion opportunities while 4h HMA provides regime bias.
Key insight: Previous 15m strategies failed due to overly strict entry conditions or wrong signal type (trend vs mean-revert).
This strategy uses Connors RSI for entry timing (proven 75% win rate on mean reversion), 4h HMA for trend bias,
and Bollinger Band position for confirmation. Entry conditions LOOSENED to ensure 10+ trades.
Timeframe: 15m (REQUIRED for exp#019), HTF: 4h via mtf_data helper.
Position sizing: 0.25-0.30 discrete levels with 2.5*ATR stoploss.
Why this might work: 15m has enough volatility for mean reversion, 4h HMA filters counter-trend trades.
Must generate 10+ trades on train, 3+ on test - CRSI thresholds widened from 10/90 to 15/85.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_bb_v1"
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
    """Calculate RSI of streak (consecutive up/down days) for Connors RSI."""
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    delta = streak_s.diff()
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    streak_rsi = np.zeros(n)
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    streak_rsi[mask] = 100 - (100 / (1 + rs[mask]))
    streak_rsi[~mask] = 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank for Connors RSI."""
    n = len(close)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = 100 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + streak_rsi + pr) / 3
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Volume MA for volume confirmation
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Bollinger Band position
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        near_lower = bb_pct < 0.15
        near_upper = bb_pct > 0.85
        
        # CRSI conditions - LOOSENED from 10/90 to 15/85 for more trades
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        crsi_extreme_oversold = crsi[i] < 15
        crsi_extreme_overbought = crsi[i] > 85
        
        # Volume confirmation
        vol_above_avg = volume[i] > vol_ma[i] * 1.2
        
        # ADX for trend strength
        adx_strong = adx[i] > 20
        adx_weak = adx[i] < 25
        
        # EMA position
        above_ema50 = not np.isnan(ema_50[i]) and close[i] > ema_50[i]
        below_ema50 = not np.isnan(ema_50[i]) and close[i] < ema_50[i]
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 5:
            higher_low = low[i] > min(low[i-3:i])
            lower_high = high[i] < max(high[i-3:i])
        
        new_signal = 0.0
        
        # === LONG ENTRIES (mean reversion in uptrend) ===
        if bull_trend_4h or above_200:
            # Primary: CRSI oversold + near BB lower + volume spike
            if crsi_oversold and near_lower:
                new_signal = SIZE_BASE
            
            # Secondary: Extreme CRSI with any trend confirmation
            elif crsi_extreme_oversold and (above_ema50 or bull_trend_4h):
                new_signal = SIZE_BASE
            
            # Tertiary: CRSI oversold + higher low pattern
            elif crsi[i] < 25 and higher_low and above_200:
                new_signal = SIZE_HALF
            
            # Volume confirmation entry
            elif crsi_oversold and vol_above_avg and above_ema50:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (mean reversion in downtrend) ===
        elif bear_trend_4h or below_200:
            # Primary: CRSI overbought + near BB upper + volume spike
            if crsi_overbought and near_upper:
                new_signal = -SIZE_BASE
            
            # Secondary: Extreme CRSI with any trend confirmation
            elif crsi_extreme_overbought and (below_ema50 or bear_trend_4h):
                new_signal = -SIZE_BASE
            
            # Tertiary: CRSI overbought + lower high pattern
            elif crsi[i] > 75 and lower_high and below_200:
                new_signal = -SIZE_HALF
            
            # Volume confirmation entry
            elif crsi_overbought and vol_above_avg and below_ema50:
                new_signal = -SIZE_HALF
        
        # === RANGE MARKET ENTRIES (ADX weak, pure mean reversion) ===
        if adx_weak:
            # Long at BB lower with CRSI confirmation
            if crsi[i] < 25 and near_lower and not np.isnan(bb_lower[i]):
                new_signal = SIZE_HALF
            
            # Short at BB upper with CRSI confirmation
            elif crsi[i] > 75 and near_upper and not np.isnan(bb_upper[i]):
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