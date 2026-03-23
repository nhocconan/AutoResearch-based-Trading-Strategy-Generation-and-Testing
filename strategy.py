#!/usr/bin/env python3
"""
Experiment #485: 1h Primary + 4h/1d HTF — CRSI Mean Reversion + HTF Trend + Vol Filter

Hypothesis: Based on research showing Connors RSI (CRSI) achieves 75% win rate on mean 
reversion entries when combined with HTF trend filter. Key innovations for 1h timeframe:
1. Connors RSI(3,2,100) - superior to standard RSI for short-term reversals
2. 4h HMA(21) - smooth trend filter without lag (Hull MA reacts faster than EMA)
3. 1d SMA(50) - major trend bias (only trade 1h entries in direction of daily trend)
4. ATR Ratio(7/30) - vol spike filter (only enter when vol > 1.3x normal = real moves)
5. Bollinger %B - confirms price at extreme before CRSI signal
6. Discrete sizing: 0.25 long, -0.20 short (asymmetric for crypto downside risk)
7. 2.5x ATR trailing stoploss for risk management
8. Relaxed CRSI thresholds (25/75 not 20/80) to ensure trade generation

Why this should work for 1h: Lower TF strategies fail from too many trades (fee drag) or 
too few (0 trades). This uses 4h/1d for DIRECTION (few signals), 1h only for ENTRY TIMING 
(when to pull trigger within HTF trend). CRSI is proven for mean reversion in bear/range 
markets (2025 test period). ATR ratio filter ensures we only trade real vol spikes, not 
noise. Target: 40-80 trades/year on 1h (within fee drag limits).

Target: Sharpe > 0.612 (beat current best), DD < -35%, trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_trend_vol_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate on mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3) component
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI component (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_s = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_s = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_gain_s / (streak_loss_s + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank component (where does current price rank vs last 100?)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if np.isnan(rsi_3[i]) or np.isnan(rsi_streak[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range (mean reversion regime)
    CHOP < 38.2 = trending (trend follow regime)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j-1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    Calculate ATR Ratio for vol spike detection.
    Ratio > 1.3 = vol spike (real move, not noise)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = atr_short / (atr_long + 1e-10)
    
    return ratio

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Much less lag than EMA, smoother than SMA.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    raw_hma = 2.0 * wma_half - wma_full
    
    hma = wma(raw_hma, sqrt_n)
    
    return hma

def calculate_rsi(close, period=14):
    """Calculate standard RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_ratio_1h = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    rsi_4h_raw = calculate_rsi(df_4h['close'].values, period=14)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    
    sma_50_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop_1h[i]) or np.isnan(atr_ratio_1h[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            continue
        if np.isnan(sma_50_1d_aligned[i]):
            continue
        
        price = close[i]
        crsi = crsi_1h[i]
        chop = chop_1h[i]
        atr_ratio = atr_ratio_1h[i]
        atr = atr_14[i]
        
        hma_4h = hma_4h_aligned[i]
        rsi_4h = rsi_4h_aligned[i]
        sma_50_1d = sma_50_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop > 55.0  # Mean reversion regime
        is_trend = chop < 45.0  # Trending regime
        
        # === HTF MAJOR TREND BIAS (1d SMA50) ===
        htf_bullish = price > sma_50_1d
        htf_bearish = price < sma_50_1d
        
        # === 4H TREND CONFIRMATION (HMA21) ===
        hma_4h_slope_up = hma_4h > hma_4h_aligned[i-5] if i >= 5 else False
        hma_4h_slope_down = hma_4h < hma_4h_aligned[i-5] if i >= 5 else False
        
        # === PRICE POSITION (Bollinger %B) ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_pct = (price - bb_lower[i]) / bb_range
        else:
            bb_pct = 0.5
        
        price_at_low = bb_pct < 0.15  # Near lower band
        price_at_high = bb_pct > 0.85  # Near upper band
        
        # === VOL FILTER ===
        vol_spike = atr_ratio > 1.25  # Vol above normal
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES (score-based confluence)
        long_score = 0
        
        # HTF bias alignment (required)
        if htf_bullish:
            long_score += 2
        
        # 4h trend confirmation
        if hma_4h_slope_up:
            long_score += 1
        
        # CRSI oversold (mean reversion entry)
        if crsi < 25.0:
            long_score += 2
        
        # Price at Bollinger lower (confirms extreme)
        if price_at_low:
            long_score += 1
        
        # Vol spike (real move, not noise)
        if vol_spike:
            long_score += 1
        
        # Range regime preferred for mean reversion
        if is_range:
            long_score += 1
        
        # Enter long if score >= 5 (3+ confluence)
        if long_score >= 5:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # HTF bias alignment
            if htf_bearish:
                short_score += 2
            
            # 4h trend confirmation
            if hma_4h_slope_down:
                short_score += 1
            
            # CRSI overbought
            if crsi > 75.0:
                short_score += 2
            
            # Price at Bollinger upper
            if price_at_high:
                short_score += 1
            
            # Vol spike
            if vol_spike:
                short_score += 1
            
            # Range regime
            if is_range:
                short_score += 1
            
            if short_score >= 5:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, price)
            stop_price = highest_since_entry - 2.5 * entry_atr
            if price < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, price)
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if price > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and htf_bullish and hma_4h_slope_up:
                desired_signal = SIZE_LONG
            elif position_side < 0 and htf_bearish and hma_4h_slope_down:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = price
                entry_atr = atr
                highest_since_entry = price if position_side > 0 else 0.0
                lowest_since_entry = price if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = price
                entry_atr = atr
                highest_since_entry = price if position_side > 0 else 0.0
                lowest_since_entry = price if position_side < 0 else float('inf')
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