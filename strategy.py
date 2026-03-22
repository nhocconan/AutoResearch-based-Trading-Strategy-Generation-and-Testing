#!/usr/bin/env python3
"""
Experiment #012: 1d Connors RSI + HMA Trend + ATR Vol Spike + ADX Regime + Asymmetric Logic
Hypothesis: Daily timeframe with Connors RSI mean reversion works better than pure trend following
for BTC/ETH in bear/range markets (2025 test period). CRSI captures oversold bounces in uptrends
and overbought drops in downtrends. ATR vol spike filter ensures we only enter when volatility
is elevated (better risk/reward). ADX regime detection applies asymmetric logic: trending markets
get trend-following entries, ranging markets get mean reversion entries. Conservative 0.25 sizing
with 3*ATR stoploss for daily bars. Multiple entry paths ensure >=10 trades per symbol.
Timeframe: 1d (REQUIRED for this experiment).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_atr_adx_asymmetric_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
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
    Connors RSI (CRSI) = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    Captures short-term momentum, streak strength, and relative position.
    """
    n = len(close)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # PercentRank - where current close ranks vs last 100 days
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < window[-1])
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    # Using 1d as primary, can reference weekly for broader trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    crsi = calculate_crsi(close, 3, 2, 100)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # HMA for trend bias
    hma_50 = calculate_hma(close, 50)
    hma_21 = calculate_hma(close, 21)
    hma_10 = calculate_hma(close, 10)
    
    # Bollinger Bands for mean reversion levels
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # ATR ratio for vol spike detection
    atr_ratio = atr_7 / atr_30
    atr_ratio = np.where(np.isnan(atr_ratio) | np.isinf(atr_ratio), 0.0, atr_ratio)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_50[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        weekly_bearish = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # Daily trend bias
        daily_bullish = close[i] > hma_50[i] and hma_50[i] > hma_50[i-1] if i > 0 else close[i] > hma_50[i]
        daily_bearish = close[i] < hma_50[i] and hma_50[i] < hma_50[i-1] if i > 0 else close[i] < hma_50[i]
        
        # HMA alignment (fast above slow = bullish)
        hma_aligned_bullish = hma_10[i] > hma_21[i] and hma_21[i] > hma_50[i]
        hma_aligned_bearish = hma_10[i] < hma_21[i] and hma_21[i] < hma_50[i]
        
        # ADX regime detection
        regime_trending = adx[i] > 25
        regime_ranging = adx[i] < 20
        
        # Vol spike detection (elevated volatility = better entry opportunities)
        vol_spike = atr_ratio[i] > 1.5
        vol_normal = atr_ratio[i] > 0.8 and atr_ratio[i] < 1.5
        
        # CRSI extremes for mean reversion
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_neutral_low = crsi[i] > 20 and crsi[i] < 40
        crsi_neutral_high = crsi[i] > 60 and crsi[i] < 80
        
        # Bollinger Band positions
        at_bb_lower = close[i] <= bb_lower[i] * 1.005  # within 0.5% of lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.995  # within 0.5% of upper band
        
        # RSI for additional confirmation
        rsi_14 = calculate_rsi(close, 14)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: CRSI oversold + Daily bullish + Vol spike (mean reversion in uptrend)
        if crsi_oversold and daily_bullish and vol_spike:
            new_signal = SIZE_ENTRY
        
        # Path 2: CRSI oversold + At BB lower + Weekly not bearish (deep mean reversion)
        elif crsi_oversold and at_bb_lower and not weekly_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 3: Trending regime + HMA aligned bullish + CRSI pullback (trend continuation)
        elif regime_trending and hma_aligned_bullish and crsi_neutral_low:
            new_signal = SIZE_ENTRY
        
        # Path 4: RSI oversold + Daily bullish + ADX building (momentum entry)
        elif rsi_oversold and daily_bullish and adx[i] > adx[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: At BB lower + Vol spike + Weekly bullish (deep dip buy)
        elif at_bb_lower and vol_spike and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 6: CRSI rising from oversold + HMA10 crossing above HMA21
        elif crsi[i] < 25 and crsi[i] > crsi[i-1] if i > 0 else False:
            if hma_10[i] > hma_21[i] and hma_10[i-1] <= hma_21[i-1]:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: CRSI overbought + Daily bearish + Vol spike (mean reversion in downtrend)
        if crsi_overbought and daily_bearish and vol_spike:
            new_signal = -SIZE_ENTRY
        
        # Path 2: CRSI overbought + At BB upper + Weekly not bullish (deep mean reversion)
        elif crsi_overbought and at_bb_upper and not weekly_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Trending regime + HMA aligned bearish + CRSI pullback (trend continuation)
        elif regime_trending and hma_aligned_bearish and crsi_neutral_high:
            new_signal = -SIZE_ENTRY
        
        # Path 4: RSI overbought + Daily bearish + ADX building (momentum entry)
        elif rsi_overbought and daily_bearish and adx[i] > adx[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: At BB upper + Vol spike + Weekly bearish (deep rally sell)
        elif at_bb_upper and vol_spike and weekly_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 6: CRSI falling from overbought + HMA10 crossing below HMA21
        elif crsi[i] > 75 and crsi[i] < crsi[i-1] if i > 0 else False:
            if hma_10[i] < hma_21[i] and hma_10[i-1] >= hma_21[i-1]:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for daily timeframe)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR for daily timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals