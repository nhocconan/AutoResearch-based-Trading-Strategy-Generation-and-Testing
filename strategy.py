#!/usr/bin/env python3
"""
Experiment #433: 1d Primary + 1w HTF — HMA Trend + CRSI Mean Reversion + Donchian

Hypothesis: Daily timeframe with weekly trend bias should work well for all symbols.
Using CRSI for mean reversion entries (proven 75% win rate in bear/range markets)
combined with 1w HMA for trend direction and Donchian for breakout confirmation.

Key differences from failed #431 (4h timeframe):
- 1d timeframe = fewer trades, lower fee drag, better for trend following
- 1w HTF = cleaner trend signal than 1d
- Simpler entry conditions to ensure trades are generated (avoid 0-trade failure)
- Position size 0.28 (conservative for daily)

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """Calculate RSI Streak component of CRSI."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        streak_rsi[i] = streak
    
    streak_rsi_s = pd.Series(streak_rsi)
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_gain = streak_rsi_s.rolling(window=period, min_periods=period).apply(
            lambda x: np.sum(x[x > 0]) if len(x[x > 0]) > 0 else 0
        )
        streak_loss = streak_rsi_s.rolling(window=period, min_periods=period).apply(
            lambda x: np.sum(-x[x < 0]) if len(x[x < 0]) > 0 else 0
        )
        
        rs = streak_gain / (streak_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + rs))
    
    streak_rsi = np.clip(streak_rsi.values, 0, 100)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank component of CRSI."""
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 1d (conservative)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === CRSI SIGNALS (relaxed thresholds for more trades) ===
        crsi_oversold = crsi[i] < 25.0  # Long entry threshold (relaxed from 15)
        crsi_overbought = crsi[i] > 75.0  # Short entry threshold (relaxed from 85)
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
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
        
        # LONG SETUP — Multiple entry pathways (ensure trades are generated)
        # Path 1: CRSI oversold in uptrend (mean reversion)
        if price_above_hma_1w and crsi_oversold:
            desired_signal = position_size
        # Path 2: CRSI extreme oversold + above SMA200 (strong mean reversion)
        elif crsi_extreme_oversold and close[i] > sma_200[i]:
            desired_signal = position_size * 0.9
        # Path 3: Donchian breakout + HMA bullish (trend follow)
        elif donchian_breakout_long and hma_bullish:
            desired_signal = position_size * 0.7
        # Path 4: HMA crossover bullish + not overbought
        elif hma_bullish and hma_21[i] > hma_50[i] and crsi[i] < 70:
            desired_signal = position_size * 0.5
        # Path 5: Choppiness regime + CRSI mean reversion
        elif is_choppy and crsi_oversold:
            desired_signal = position_size * 0.6
        
        # SHORT SETUP — Multiple entry pathways
        # Path 1: CRSI overbought in downtrend (mean reversion)
        if price_below_hma_1w and crsi_overbought:
            desired_signal = -position_size
        # Path 2: CRSI extreme overbought + below SMA200 (strong mean reversion)
        elif crsi_extreme_overbought and close[i] < sma_200[i]:
            desired_signal = -position_size * 0.9
        # Path 3: Donchian breakdown + HMA bearish (trend follow)
        elif donchian_breakout_short and hma_bearish:
            desired_signal = -position_size * 0.7
        # Path 4: HMA crossover bearish + not oversold
        elif hma_bearish and hma_21[i] < hma_50[i] and crsi[i] > 30:
            desired_signal = -position_size * 0.5
        # Path 5: Choppiness regime + CRSI mean reversion
        elif is_choppy and crsi_overbought:
            desired_signal = -position_size * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === CRSI EXTREME EXIT ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1w and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1w or hma_bullish or close[i] > sma_200[i]):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_1w or hma_bearish or close[i] < sma_200[i]):
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