#!/usr/bin/env python3
"""
Experiment #732: 12h Primary + 1d/1w HTF — Dual Regime Strategy

Hypothesis: 12h timeframe with dual regime detection (trend vs range) can beat
the current 4h best (Sharpe=0.612). Key insights from 490+ failed strategies:

1. Choppiness Index (CHOP) is the best regime filter for crypto
   - CHOP > 61.8 = ranging market → use mean reversion (CRSI)
   - CHOP < 38.2 = trending market → use trend following (Donchian/HMA)
   
2. Connors RSI (CRSI) works better than standard RSI for mean reversion
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Entry: CRSI < 10 (long) or CRSI > 90 (short)
   
3. 1d/1w HMA provides strong trend bias without lag
   - Only trade in direction of HTF trend for trend-following regime
   - Mean reversion can trade against HTF trend in ranging markets

4. 12h timeframe targets 20-50 trades/year (low fee drag)
   - Use BASE_SIZE = 0.28 (conservative for higher TF)
   - ATR stoploss at 2.5x to avoid premature exits

5. Multiple entry paths ensure trade frequency (learned from #728-731 zero trades)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (proven higher TF works better for crypto)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close):
    """
    Connors RSI (CRSI) - proven mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - CRSI < 10: oversold → long
    - CRSI > 90: overbought → short
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < 100:
        return crsi
    
    # RSI(3) - very short term momentum
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(2, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 - (100 / (1 + streak[i]))
        elif streak[i] < 0:
            streak_rsi[i] = 100 / (1 + abs(streak[i]))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100) - where current return ranks vs last 100
    pct_rank = np.full(n, np.nan)
    for i in range(100, n):
        returns = np.diff(close[i-99:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            pct_rank[i] = rank * 100
    
    # Combine components
    valid_mask = ~np.isnan(rsi3) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[valid_mask] = (rsi3[valid_mask] + streak_rsi[valid_mask] + pct_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending.
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Ranging market (mean reversion works)
    - CHOP < 38.2: Trending market (trend following works)
    - 38.2 - 61.8: Transition zone
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Rolling calculations
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    crsi_12h = calculate_crsi(close)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    hma_50 = calculate_hma(close, period=50)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(crsi_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(donch_upper[i]) or np.isnan(hma_50[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = chop_12h[i] > 61.8
        is_trending = chop_12h[i] < 38.2
        # Transition zone: 38.2 - 61.8 (use trend following with caution)
        
        # === TREND BIAS (1d and 1w HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend when both 1d and 1w agree
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        
        # REGIME 1: RANGING MARKET - Mean Reversion with CRSI
        if is_ranging:
            # CRSI deeply oversold (strong mean reversion signal)
            if crsi_12h[i] < 15:
                long_signal = True
            # RSI oversold + near Donchian lower (support bounce)
            elif rsi_12h[i] < 30 and close[i] < donch_lower[i-1] * 1.02:
                long_signal = True
        
        # REGIME 2: TRENDING MARKET - Trend Following
        if is_trending or (38.2 <= chop_12h[i] <= 61.8):
            # Strong bullish trend + pullback to HMA50
            if strong_bullish and close[i] < hma_50[i] * 1.01 and close[i] > hma_50[i] * 0.98:
                long_signal = True
            # Donchian breakout + bullish trend
            elif close[i] > donch_upper[i-1] and trend_1d_bullish:
                long_signal = True
            # RSI pullback in uptrend (buy the dip)
            elif trend_1d_bullish and rsi_12h[i] < 45 and above_sma200:
                long_signal = True
        
        # REGIME 3: TRANSITION - Conservative entries
        if 38.2 <= chop_12h[i] <= 61.8:
            # Very oversold CRSI + above 1d HMA
            if crsi_12h[i] < 10 and trend_1d_bullish:
                long_signal = True
        
        if long_signal:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        
        # REGIME 1: RANGING MARKET - Mean Reversion with CRSI
        if is_ranging:
            # CRSI deeply overbought (strong mean reversion signal)
            if crsi_12h[i] > 85:
                short_signal = True
            # RSI overbought + near Donchian upper (resistance rejection)
            elif rsi_12h[i] > 70 and close[i] > donch_upper[i-1] * 0.98:
                short_signal = True
        
        # REGIME 2: TRENDING MARKET - Trend Following
        if is_trending or (38.2 <= chop_12h[i] <= 61.8):
            # Strong bearish trend + bounce to HMA50
            if strong_bearish and close[i] > hma_50[i] * 0.99 and close[i] < hma_50[i] * 1.02:
                short_signal = True
            # Donchian breakdown + bearish trend
            elif close[i] < donch_lower[i-1] and trend_1d_bearish:
                short_signal = True
            # RSI bounce in downtrend (sell the rip)
            elif trend_1d_bearish and rsi_12h[i] > 55 and below_sma200:
                short_signal = True
        
        # REGIME 3: TRANSITION - Conservative entries
        if 38.2 <= chop_12h[i] <= 61.8:
            # Very overbought CRSI + below 1d HMA
            if crsi_12h[i] > 90 and trend_1d_bearish:
                short_signal = True
        
        if short_signal:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with stronger trend (1w HMA)
        if long_signal and short_signal:
            if trend_1w_bullish:
                desired_signal = current_size
            elif trend_1w_bearish:
                desired_signal = -current_size
            else:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC — Maintain position if regime/trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if ranging (CRSI not overbought) or trending bullish
                if (is_ranging and crsi_12h[i] < 70) or (trend_1d_bullish and rsi_12h[i] < 75):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if ranging (CRSI not oversold) or trending bearish
                if (is_ranging and crsi_12h[i] > 30) or (trend_1d_bearish and rsi_12h[i] > 25):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought (mean reversion complete) or trend reverses
            if (is_ranging and crsi_12h[i] > 75) or (trend_1d_bearish and rsi_12h[i] > 60):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold (mean reversion complete) or trend reverses
            if (is_ranging and crsi_12h[i] < 25) or (trend_1d_bullish and rsi_12h[i] < 40):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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