#!/usr/bin/env python3
"""
Experiment #042: 12h Primary + 1d/1w HTF — Dual Regime with Connors RSI + Donchian

Hypothesis: 12h timeframe is proven to work (Exp #032 got Sharpe=0.479). 
Key improvements over failed experiments:
1) LOOSE entry thresholds (CRSI 25/75 not 10/90, CHOP 55/45 not 61.8/38.2)
2) Multiple fallback entry triggers to ensure trade generation
3) 1w HMA for ultra-long trend bias (avoids counter-trend in bear markets)
4) ATR(14) 2.5x trailing stop for risk management
5) Position size 0.30 (discrete, within safe 0.20-0.35 range)

Why 12h works:
- 20-50 trades/year = low fee drag (1-2.5% annually)
- Less noise than 4h/1h, more signals than 1d
- Proven in Exp #032 (Sharpe=0.479, Return=+104.8%)

Entry logic:
- Ranging (CHOP>55): Connors RSI mean reversion + BB extremes
- Trending (CHOP<45): Donchian breakout + HMA alignment
- Fallback: HMA crossover + RSI confirmation (ensures trades)

Position size: 0.30 | Stoploss: 2.5*ATR trailing | leverage=1.0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_donchian_dualhtf_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    return hma.values

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
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
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2) - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = sum(1 for j in range(i-streak_period+1, i+1) if streak[j] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    
    # PercentRank(100) - where current price ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = sum(1 for p in window if p < close[i])
        percent_rank[i] = (rank / rank_period) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short.values + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for medium-term bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for long-term bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    hma_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):  # Warmup for all indicators including CRSI rank_period
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(donchian_upper[i]) or np.isnan(hma_21[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both 1d and 1w agree
        strong_bullish = price_above_hma_1d and price_above_hma_1w
        strong_bearish = price_below_hma_1d and price_below_hma_1w
        
        # === CHOPPINESS REGIME (with hysteresis) ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market (LOOSE threshold)
        is_trending = chop_value < 45.0  # Trend market (with hysteresis gap)
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # Very oversold
        crsi_overbought = crsi[i] > 75.0  # Very overbought
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        crsi_neutral = 35.0 < crsi[i] < 65.0
        
        # === STANDARD RSI (backup filter) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_squeeze = (bb_upper[i] - bb_lower[i]) < (bb_upper[i-20] - bb_lower[i-20]) * 0.8 if i > 20 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === HMA TREND ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        hma_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with Connors RSI ---
        if is_ranging:
            # Long: CRSI oversold + BB support + any bullish confirmation
            if crsi_oversold or (rsi_oversold and price_below_bb_lower):
                if crsi_rising or rsi_rising or price_above_hma_1d:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + BB resistance + any bearish confirmation
            elif crsi_overbought or (rsi_overbought and price_above_bb_upper):
                if crsi_falling or rsi_falling or price_below_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Donchian ---
        elif is_trending:
            # Long: Donchian breakout + HMA bullish + HTF confirms
            if donchian_breakout_long and hma_bullish:
                if hma_slope_up or strong_bullish or price_above_hma_1d:
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + HMA bearish + HTF confirms
            elif donchian_breakout_short and hma_bearish:
                if hma_slope_down or strong_bearish or price_below_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK 1: HMA crossover (ensures trades in neutral regime) ---
        if new_signal == 0.0 and not is_ranging and not is_trending:
            # Long: Price crosses above HMA + RSI confirmation
            if close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]:
                if rsi_14[i] > 45 and (crsi_rising or price_above_hma_1d):
                    new_signal = POSITION_SIZE
            
            # Short: Price crosses below HMA + RSI confirmation
            elif close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]:
                if rsi_14[i] < 55 and (crsi_falling or price_below_hma_1d):
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK 2: BB mean reversion (ensures trades) ---
        if new_signal == 0.0:
            # Long: Price below BB lower + CRSI rising
            if price_below_bb_lower and crsi_rising and crsi[i] < 40:
                new_signal = POSITION_SIZE
            
            # Short: Price above BB upper + CRSI falling
            elif price_above_bb_upper and crsi_falling and crsi[i] > 60:
                new_signal = -POSITION_SIZE
        
        # --- FALLBACK 3: RSI extreme reversal (ensures trades) ---
        if new_signal == 0.0:
            # Long: RSI oversold + turning up
            if rsi_14[i] < 30 and rsi_rising:
                if price_above_hma_1w or crsi[i] < 50:
                    new_signal = POSITION_SIZE
            
            # Short: RSI overbought + turning down
            elif rsi_14[i] > 70 and rsi_falling:
                if price_below_hma_1w or crsi[i] > 50:
                    new_signal = -POSITION_SIZE
        
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
        
        # === EXIT ON STRONG REGIME CHANGE ===
        if in_position and position_side > 0:
            if strong_bearish and chop_value < 40:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if strong_bullish and chop_value < 40:
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