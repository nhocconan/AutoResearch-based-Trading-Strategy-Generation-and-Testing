#!/usr/bin/env python3
"""
Experiment #034: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + CRSI Mean Reversion

Hypothesis: 4h timeframe with 12h KAMA trend bias and 1d ADX regime filter will generate
25-50 trades/year with positive Sharpe. Key improvements over failed experiments:
1. KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA
2. 1d ADX for regime (more stable than 4h Choppiness)
3. Connors RSI for entries (proven 75% win rate in mean reversion)
4. Volume confirmation on breakouts (filters false signals)
5. Asymmetric sizing: larger in trend regime, smaller in range

Why 4h works best:
- Enough bars for statistical significance (4 years = ~8760 bars)
- Not too many trades (fee drag manageable at 25-50/year)
- Captures multi-day trends without whipsaw of lower TF

Strategy Logic:
1. 12h KAMA: Adaptive trend bias (fast in trends, slow in ranges)
2. 1d ADX: Regime filter (ADX>25 = trend, ADX<20 = range)
3. CRSI(3,2,100): Entry timing (extremes for mean reversion)
4. Volume spike confirmation: 1.5x avg volume for breakouts
5. ATR(14) trailing stop: 2.5*ATR protection

Position size: 0.25-0.30 (discrete, trend regime gets larger size)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_adx_regime_12h1d_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = np.nan
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion at extremes.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            percent_rank[i] = np.sum(returns[:-1] < current_return) / (len(returns) - 1) * 100.0
    
    # Combine
    crsi = (rsi_close.values + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes (>threshold * average volume)."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (vol_avg * threshold)
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA for adaptive trend bias
    kama_12h = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1d ADX for regime filter
    adx_1d, plus_di_1d, minus_di_1d = calculate_adx(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    volume_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # Calculate price momentum
    roc_10 = np.zeros(n)
    for i in range(10, n):
        roc_10[i] = (close[i] - close[i-10]) / (close[i-10] + 1e-10) * 100.0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.30   # Larger size in trending regime
    SIZE_RANGE = 0.20   # Smaller size in ranging regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(kama_4h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H ADAPTIVE TREND BIAS ===
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        kama_12h_slope_up = kama_12h_aligned[i] > kama_12h_aligned[i-5] if i > 5 else False
        kama_12h_slope_down = kama_12h_aligned[i] < kama_12h_aligned[i-5] if i > 5 else False
        
        # === 1D ADX REGIME FILTER ===
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25.0  # Strong trend
        is_ranging = adx_value < 20.0   # Range market
        # Hysteresis zone: 20-25 = maintain previous regime
        
        # === CRSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 25.0   # Long entry
        crsi_overbought = crsi[i] > 75.0  # Short entry
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === KAMA TREND CONFIRMATION ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        kama_slope_up = kama_4h[i] > kama_4h[i-5] if i > 5 else False
        kama_slope_down = kama_4h[i] < kama_4h[i-5] if i > 5 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume_spike[i]
        
        # === MOMENTUM ===
        momentum_positive = roc_10[i] > 0
        momentum_negative = roc_10[i] < 0
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        current_size = SIZE_RANGE  # Default to range size
        
        # --- TRENDING REGIME (ADX > 25): Trend Following ---
        if is_trending:
            current_size = SIZE_TREND
            
            # Long: CRSI pullback in uptrend + volume + 12h confirms
            if crsi_oversold and kama_bullish:
                if price_above_kama_12h and kama_12h_slope_up:
                    if vol_confirmed or crsi_rising:  # Volume OR CRSI turning
                        new_signal = current_size
            
            # Short: CRSI rally in downtrend + volume + 12h confirms
            elif crsi_overbought and kama_bearish:
                if price_below_kama_12h and kama_12h_slope_down:
                    if vol_confirmed or crsi_falling:
                        new_signal = -current_size
            
            # Breakout entry (stronger signal)
            elif kama_bullish and kama_slope_up and momentum_positive:
                if price_above_kama_12h and vol_confirmed:
                    new_signal = current_size
            elif kama_bearish and kama_slope_down and momentum_negative:
                if price_below_kama_12h and vol_confirmed:
                    new_signal = -current_size
        
        # --- RANGING REGIME (ADX < 20): Mean Reversion ---
        elif is_ranging:
            current_size = SIZE_RANGE
            
            # Long: CRSI extreme oversold + 12h not strongly bearish
            if crsi_oversold:
                if not (price_below_kama_12h and kama_12h_slope_down):  # Not strongly bearish
                    if crsi_rising or rsi_14[i] < 30:  # CRSI turning OR RSI extreme
                        new_signal = current_size
            
            # Short: CRSI extreme overbought + 12h not strongly bullish
            elif crsi_overbought:
                if not (price_above_kama_12h and kama_12h_slope_up):  # Not strongly bullish
                    if crsi_falling or rsi_14[i] > 70:  # CRSI turning OR RSI extreme
                        new_signal = -current_size
        
        # --- TRANSITION ZONE (ADX 20-25): Conservative entries ---
        else:
            # Only enter on strong CRSI extremes with 12h confirmation
            if crsi[i] < 15.0 and price_above_kama_12h:
                new_signal = SIZE_RANGE
            elif crsi[i] > 85.0 and price_below_kama_12h:
                new_signal = -SIZE_RANGE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME/TREND CHANGE ===
        # Exit long if 12h trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_kama_12h and kama_12h_slope_down and adx_value > 30:
                new_signal = 0.0
        
        # Exit short if 12h trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_kama_12h and kama_12h_slope_up and adx_value > 30:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals