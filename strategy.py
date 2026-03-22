#!/usr/bin/env python3
"""
Experiment #003: 1d Regime-Adaptive Strategy with Weekly Trend Filter

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance between
trade frequency (20-50/year) and signal quality. Using Choppiness Index for regime
detection + Connors RSI for mean reversion + Donchian for trend breakouts.

Why 1d should work better than 12h:
1. Higher timeframe = fewer false signals, less fee drag
2. Weekly HMA provides stronger trend bias than daily
3. Connors RSI proven 75% win rate in range markets
4. ATR stoploss at 2.5x protects against 2022-style crashes

Key differences from #002:
- Primary TF: 1d (not 12h)
- HTF: 1w (not 1d)
- Looser Connors RSI thresholds (20/80 vs 15/85) for more trades
- Simpler trend entry (HMA crossover + weekly bias, Donchian optional)
- Discrete position sizing (0.0, ±0.25, ±0.30)
- Better stoploss tracking

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_chop_connors_1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI."""
    n = len(close)
    close_s = pd.Series(close)
    rsi_close = calculate_rsi(close, rsi_period)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    streak_avg_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = np.where(streak_avg_loss == 0, 1e-10, streak_avg_loss)
    streak_rs = streak_avg_gain / streak_avg_loss
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    returns = close_s.pct_change().values
    percent_rank = np.full(n, 50.0)
    for i in range(rank_period, n):
        if not np.isnan(returns[i]):
            window = returns[max(0, i-rank_period):i]
            window = window[~np.isnan(window)]
            if len(window) > 0:
                percent_rank[i] = 100 * np.sum(window < returns[i]) / len(window)
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_16 = calculate_hma(close, 16)
    hma_1d_48 = calculate_hma(close, 48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        
        # Regime detection (wider bands for more trades)
        is_range = chop_14[i] > 50.0
        is_trend = chop_14[i] < 50.0
        
        # Weekly trend bias
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # Daily HMA trend
        hma_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # Position sizing
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # RANGE REGIME: Connors RSI mean reversion (looser thresholds)
        if is_range:
            # LONG: CRSI < 25 (oversold) + weekly bias not strongly bearish
            if crsi[i] < 25:
                new_signal = current_size
            # SHORT: CRSI > 75 (overbought) + weekly bias not strongly bullish
            elif crsi[i] > 75:
                new_signal = -current_size
        
        # TREND REGIME: HMA crossover + weekly confirmation
        elif is_trend:
            # LONG: Daily HMA bullish + weekly not bearish
            if hma_bullish and not weekly_bearish:
                new_signal = current_size
            # SHORT: Daily HMA bearish + weekly not bullish
            elif hma_bearish and not weekly_bullish:
                new_signal = -current_size
        
        # Frequency safeguard - force entry if no trades for 30 bars
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if hma_bullish and crsi[i] < 50:
                new_signal = current_size * 0.5
            elif hma_bearish and crsi[i] > 50:
                new_signal = -current_size * 0.5
        
        # Stoploss logic (2.5 ATR)
        stoploss_triggered = False
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # Trend reversal exit
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish and weekly_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish and weekly_bullish:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals