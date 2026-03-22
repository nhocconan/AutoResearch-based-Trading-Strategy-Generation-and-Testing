#!/usr/bin/env python3
"""
Experiment #072: 1d Regime-Adaptive Strategy with 1w HMA Trend Filter
Hypothesis: Daily timeframe reduces noise and fee impact significantly. 1w HMA provides robust long-term trend bias.
Regime-adaptive logic: trending (ADX>20) = trend follow via Donchian/EMA, ranging (ADX<20) = mean revert via CRSI/Z-score.
Connors RSI for mean reversion entries (proven 75% win rate in literature).
Donchian breakout for trend continuation signals.
Multiple entry paths to ensure 10+ trades per symbol while maintaining edge.
Position sizing: 0.25 base, 0.35 strong signals, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop on all positions.
Why this might work: 1d has fewer false signals than intraday. 1w filter avoids counter-trend trades.
Multiple entry paths ensure sufficient trade frequency. Regime detection adapts to market conditions.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_1w_hma_crsi_v2"
timeframe = "1d"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - fast RSI
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_pos = np.where(streak > 0, streak, 0)
    streak_neg = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_pos).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_neg).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    mask = avg_streak_loss > 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    streak_rsi[mask] = 100 - (100 / (1 + rs_streak[mask]))
    streak_rsi[~mask] = 100.0
    
    # Percent Rank - where current close ranks vs last 100 closes
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = rank / (rank_period - 1) * 100
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi3[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_3 = calculate_rsi(close, 3)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Donchian Channel for breakouts
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Connors RSI for mean reversion
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Z-score for mean reversion
    zscore = calculate_zscore(close, 20)
    
    # HMA on 1d for trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = long-term trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        trending_regime = adx[i] > 20
        ranging_regime = adx[i] < 20
        
        # === TREND CONDITIONS ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        hma_bullish = hma_21[i] > hma_50[i] if not np.isnan(hma_21[i]) and not np.isnan(hma_50[i]) else False
        hma_bearish = hma_21[i] < hma_50[i] if not np.isnan(hma_21[i]) and not np.isnan(hma_50[i]) else False
        
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === CONNORS RSI ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 25
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 75
        
        # === Z-SCORE ===
        zscore_oversold = not np.isnan(zscore[i]) and zscore[i] < -1.5
        zscore_overbought = not np.isnan(zscore[i]) and zscore[i] > 1.5
        
        # === RSI ===
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_mid = 40 <= rsi[i] <= 60
        
        new_signal = 0.0
        
        # === LONG ENTRY (multiple paths for trade frequency) ===
        
        # Path 1: Trending + Donchian breakout + trend alignment
        if trending_regime and bull_trend_1w:
            if breakout_long and (di_bullish or ema_bullish):
                new_signal = SIZE_STRONG
        
        # Path 2: Trending + EMA/HMA alignment
        if trending_regime and bull_trend_1w:
            if ema_bullish and hma_bullish and rsi[i] > 35 and rsi[i] < 70:
                new_signal = SIZE_BASE
        
        # Path 3: Ranging + Mean reversion (CRSI)
        if ranging_regime:
            if crsi_oversold and above_sma200:
                new_signal = SIZE_HALF
            elif zscore_oversold and rsi_oversold:
                if bull_trend_1w or above_sma200:
                    new_signal = SIZE_HALF
        
        # Path 4: Simple trend continuation (loose conditions)
        if bull_trend_1w and ema_bullish:
            if rsi_mid and di_bullish:
                new_signal = SIZE_BASE
        
        # Path 5: HMA crossover signal
        if hma_bullish and bull_trend_1w:
            if close[i] > ema_21[i] and rsi[i] > 40:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY (multiple paths for trade frequency) ===
        
        # Path 1: Trending + Donchian breakout + trend alignment
        if trending_regime and bear_trend_1w:
            if breakout_short and (di_bearish or ema_bearish):
                new_signal = -SIZE_STRONG
        
        # Path 2: Trending + EMA/HMA alignment
        if trending_regime and bear_trend_1w:
            if ema_bearish and hma_bearish and rsi[i] > 30 and rsi[i] < 65:
                new_signal = -SIZE_BASE
        
        # Path 3: Ranging + Mean reversion (CRSI)
        if ranging_regime:
            if crsi_overbought and below_sma200:
                new_signal = -SIZE_HALF
            elif zscore_overbought and rsi_overbought:
                if bear_trend_1w or below_sma200:
                    new_signal = -SIZE_HALF
        
        # Path 4: Simple trend continuation (loose conditions)
        if bear_trend_1w and ema_bearish:
            if rsi_mid and di_bearish:
                new_signal = -SIZE_BASE
        
        # Path 5: HMA crossover signal
        if hma_bearish and bear_trend_1w:
            if close[i] < ema_21[i] and rsi[i] < 60:
                new_signal = -SIZE_BASE
        
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