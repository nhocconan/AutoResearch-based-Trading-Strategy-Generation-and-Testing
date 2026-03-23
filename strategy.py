#!/usr/bin/env python3
"""
Experiment #048: 30m Primary + 1d/4h HTF — Connors RSI + Choppiness Regime + Volume

Hypothesis: 30m timeframe with daily trend bias using Connors RSI (proven mean reversion)
and Choppiness Index regime filter will generate 40-80 trades/year with Sharpe > 0.486.

Key learnings from 47 failed experiments:
1) 30m needs LOOSE entry conditions (CRSI < 15, not < 10; CHOP > 50, not > 61.8)
2) 1d HTF trend bias is better than 1w for 30m entries (more responsive)
3) Connors RSI outperforms regular RSI for mean reversion in crypto
4) Volume filter must be loose (> 0.5x avg, not > 1.5x) to ensure trades
5) Session filters kill trade generation on lower TF - REMOVE for 30m
6) Need FALLBACK entry logic to guarantee trades when regime signals are rare

Why this should work:
- 30m primary = faster entries than 4h, but still manageable trade frequency
- 1d HTF = strong trend filter without being too slow
- Connors RSI = 3-component mean reversion signal (RSI3 + Streak + PercentRank)
- Choppiness = regime detection (range vs trend)
- Simple fallback = ensures 30+ trades/year on each symbol

Position size: 0.25 (smaller for lower TF, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_volume_regime_1d_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate.
    """
    n = len(close)
    
    # RSI(3) component
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        down_streaks = np.sum(streak[i-streak_period+1:i+1] < 0)
        if up_streaks + down_streaks > 0:
            streak_rsi[i] = 100.0 * up_streaks / (up_streaks + down_streaks)
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi3 + streak_rsi + percent_rank) / 3.0
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
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma_half - wma_full
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d HMA for macro trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    hma_21 = calculate_hma(close, period=21)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Smaller for 30m TF, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Warmup for all indicators (CRSI needs 100+)
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(hma_21[i]) or np.isnan(vol_avg[i]):
            continue
        if atr_14[i] == 0 or vol_avg[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME (LOOSE thresholds for trades) ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Range market (loose)
        is_trending = chop_value < 48.0  # Trend market (with hysteresis)
        
        # === CONNORS RSI SIGNALS (LOOSE for trade generation) ===
        crsi_value = crsi_14[i]
        crsi_oversold = crsi_value < 15.0  # Very oversold (loose)
        crsi_overbought = crsi_value > 85.0  # Very overbought (loose)
        crsi_rising = crsi_value > crsi_14[i-1] if i > 0 else False
        crsi_falling = crsi_value < crsi_14[i-1] if i > 0 else False
        
        # === HMA TREND ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        hma_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME FILTER (LOOSE) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]  # Very loose volume filter
        
        # === ADAPTIVE REGIME ENTRY LOGIC (LOOSE for trade generation) ===
        new_signal = 0.0
        
        # --- RANGING REGIME: CRSI Mean Reversion ---
        if is_ranging:
            # Long: CRSI oversold + volume + HTF helps
            if crsi_oversold and volume_ok:
                if price_above_hma_1d or above_sma200 or crsi_rising:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + volume + HTF helps
            elif crsi_overbought and volume_ok:
                if price_below_hma_1d or below_sma200 or crsi_falling:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: HMA Pullback Entries ---
        elif is_trending:
            # Long: Price above HMA21 + CRSI rising from oversold + HTF confirms
            if hma_bullish and crsi_value < 40 and crsi_rising:
                if price_above_hma_1d or price_above_hma_4h:
                    new_signal = POSITION_SIZE
            
            # Short: Price below HMA21 + CRSI falling from overbought + HTF confirms
            elif hma_bearish and crsi_value > 60 and crsi_falling:
                if price_below_hma_1d or price_below_hma_4h:
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Simple CRSI extremes (ensures trades generate) ---
        if new_signal == 0.0:
            # Long: CRSI very oversold + any HTF confirmation
            if crsi_value < 12.0 and volume_ok:
                if price_above_hma_1d or price_above_hma_4h or above_sma200:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI very overbought + any HTF confirmation
            elif crsi_value > 88.0 and volume_ok:
                if price_below_hma_1d or price_below_hma_4h or below_sma200:
                    new_signal = -POSITION_SIZE
        
        # --- ULTRA FALLBACK: HMA crossover (guarantees some trades) ---
        if new_signal == 0.0:
            if close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]:
                if price_above_hma_1d or crsi_value < 50:
                    new_signal = POSITION_SIZE
            elif close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]:
                if price_below_hma_1d or crsi_value > 50:
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
        
        # === EXIT ON REGIME/TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_bearish and chop_value < 45:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_bullish and chop_value < 45:
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