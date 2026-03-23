#!/usr/bin/env python3
"""
Experiment #402: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + HMA + Donchian

Hypothesis: 12h timeframe with Choppiness Index regime detection + Connors RSI entries
will generate 20-50 trades/year with better Sharpe than 4h strategies.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven 75% win rate in research notes (ETH Sharpe +0.923 with CHOP+CRSI)
   - Long: CRSI<10, Short: CRSI>90
2. Choppiness Index regime: >61.8 = range (mean revert), <38.2 = trend
3. HMA(21/63) for trend direction on 12h
4. 1d HTF HMA for overall bias, 1w HTF for major trend filter
5. Donchian(20) breakout confirmation for entry timing
6. ATR(14) trailing stoploss (2.5x)
7. Discrete position sizing: 0.0, ±0.28 to minimize fee churn

Why 12h should work better than 4h:
- Less noise, fewer false signals
- Target 20-50 trades/year = ~1-2% fee drag vs 2-4% on 4h
- Historical success: #392 (12h) Sharpe=0.119, #396 (12h) Sharpe=0.017
- Need to beat current best Sharpe=0.612

Target: Sharpe > 0.612, 25-50 trades/year, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: percentage of past closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum((streak[i-streak_period+1:i+1] > 0) * streak_abs[i-streak_period+1:i+1])
        down_streaks = np.sum((streak[i-streak_period+1:i+1] < 0) * streak_abs[i-streak_period+1:i+1])
        if up_streaks + down_streaks > 0:
            streak_rsi[i] = 100.0 * up_streaks / (up_streaks + down_streaks)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_lower / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_close + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_63 = calculate_hma(close, 63)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Calculate and align HTF HMAs for bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (target 20-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_63[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market (mean revert)
        is_trending = chop[i] < 38.2  # Trend market (trend follow)
        
        # === HTF BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (12h HMA crossover) ===
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25.0
        adx_weak = adx_14[i] < 20.0
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Long entry
        crsi_overbought = crsi[i] > 85.0  # Short entry
        
        # === DONCHIAN BREAKOUT ===
        donchian_long = close[i] > donchian_upper[i-1]
        donchian_short = close[i] < donchian_lower[i-1]
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple confluence paths (relaxed for more trades)
        long_bias = price_above_hma_1d or price_above_hma_1w  # Either HTF bullish
        
        if long_bias:
            if is_choppy:
                # Mean reversion in range: CRSI oversold
                if crsi_oversold:
                    desired_signal = position_size
            elif is_trending and hma_bullish:
                # Trend following: HMA bullish + CRSI not extreme
                if crsi[i] < 70.0:
                    desired_signal = position_size
            elif hma_bullish and crsi_oversold:
                # Pullback in uptrend with CRSI confirmation
                desired_signal = position_size
            elif donchian_long and (hma_bullish or adx_strong):
                # Donchian breakout with trend confirmation
                desired_signal = position_size
            elif crsi[i] < 30.0 and hma_bullish:
                # Moderate CRSI oversold in uptrend
                desired_signal = position_size
        
        # SHORT SETUP - Multiple confluence paths
        short_bias = price_below_hma_1d or price_below_hma_1w  # Either HTF bearish
        
        if short_bias:
            if is_choppy:
                # Mean reversion in range: CRSI overbought
                if crsi_overbought:
                    desired_signal = -position_size
            elif is_trending and hma_bearish:
                # Trend following: HMA bearish + CRSI not extreme
                if crsi[i] > 30.0:
                    desired_signal = -position_size
            elif hma_bearish and crsi_overbought:
                # Rally in downtrend with CRSI confirmation
                desired_signal = -position_size
            elif donchian_short and (hma_bearish or adx_strong):
                # Donchian breakdown with trend confirmation
                desired_signal = -position_size
            elif crsi[i] > 70.0 and hma_bearish:
                # Moderate CRSI overbought in downtrend
                desired_signal = -position_size
        
        # === STOPLOSS CHECK (ATR trailing) ===
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
        
        # === CRSI EXIT (extreme reached - take profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1d and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1d or price_above_hma_1w):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_1d or price_below_hma_1w):
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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