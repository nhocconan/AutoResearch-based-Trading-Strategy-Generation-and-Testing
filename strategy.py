#!/usr/bin/env python3
"""
Experiment #707: 1d Primary + 1w HTF — Choppiness + Connors RSI + HMA Trend + ADX

Hypothesis: Daily timeframe with weekly trend bias captures major moves while avoiding noise.
Choppiness Index distinguishes range vs trend regimes. Connors RSI provides superior mean-reversion
signals vs standard RSI. HMA(21) on weekly provides strong trend filter. ADX confirms trend strength.
Dual regime: mean-revert in chop (CHOP>61.8), trend-follow when ADX>25 + breakout.

Why this should work:
1. 1d TF reduces noise vs lower TFs — fewer false signals
2. Connors RSI has 75% win rate in literature for mean reversion
3. Choppiness Index >61.8 reliably identifies ranges (Ehlers research)
4. Weekly HMA filter prevents counter-trend trades that destroyed 2022 performance
5. ADX filter prevents breakout trades in weak trends (common failure mode)
6. Target 20-50 trades/year on 1d = low fee drag

Key differences from #697 (which got Sharpe=-0.231):
1. Looser CRSI thresholds (25/75 not 15/85) for MORE trades
2. Add ADX confirmation for trend mode (prevents weak breakouts)
3. Simpler hold logic — only exit on stoploss or clear trend reversal
4. Reduced signal churn — discrete levels only (0.0, ±0.25, ±0.30)
5. Earlier entry buffer (120 not 150) to capture more trades

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_adx_donchian_1w_v2"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) by E.W. Dreiss
    Measures market choppiness vs trending
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Larry Connors
    CRSI < 25 = oversold (long signal)
    CRSI > 75 = overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
    avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            pct_rank[i] = 100 * np.sum(returns < current_return) / len(returns)
    
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(rsi_streak) & ~np.isnan(pct_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + rsi_streak[valid_mask] + pct_rank[valid_mask]) / 3
    
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    df_1w = get_htf_data(prices, '1w')
    
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_1d = calculate_atr(high, low, close, period=14)
    adx_1d = calculate_adx(high, low, close, period=14)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(120, n):
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(atr_1d[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(adx_1d[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        trend_bullish = close[i] > hma_1w_aligned[i]
        trend_bearish = close[i] < hma_1w_aligned[i]
        
        adx_strong = adx_1d[i] > 25
        adx_weak = adx_1d[i] < 20
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        if is_choppy or adx_weak:
            if crsi[i] < 30 and trend_bullish:
                desired_signal = current_size
            elif crsi[i] > 70 and trend_bearish:
                desired_signal = -current_size
            elif crsi[i] < 25:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 75:
                desired_signal = -REDUCED_SIZE
        
        elif is_trending and adx_strong:
            if close[i] > donchian_upper[i] and trend_bullish:
                desired_signal = current_size
            elif close[i] < donchian_lower[i] and trend_bearish:
                desired_signal = -current_size
            elif close[i] > donchian_upper[i]:
                desired_signal = REDUCED_SIZE
            elif close[i] < donchian_lower[i]:
                desired_signal = -REDUCED_SIZE
        
        else:
            if crsi[i] < 25 and trend_bullish:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 75 and trend_bearish:
                desired_signal = -REDUCED_SIZE
        
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if crsi[i] < 80 and trend_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if crsi[i] > 20 and trend_bearish:
                    desired_signal = -BASE_SIZE
        
        if in_position and position_side > 0:
            if crsi[i] > 85:
                desired_signal = 0.0
            elif close[i] < hma_1w_aligned[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if crsi[i] < 15:
                desired_signal = 0.0
            elif close[i] > hma_1w_aligned[i]:
                desired_signal = 0.0
        
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals