#!/usr/bin/env python3
"""
Experiment #050: 1h Primary + 4h/12h HTF — CRSI Mean Reversion + Regime Adaptive

Hypothesis: 1h timeframe with 12h trend bias + 4h regime filter + 1h Connors RSI entries
will generate 40-80 trades/year with Sharpe > 0.486 (beat current best).

Key learnings from 49 experiments:
1) 1h strategies fail with Sharpe=0.000 when entry conditions too strict (exp#040, #045)
2) CRSI thresholds MUST be loose (20/80 not 10/90) to ensure trade generation
3) Multi-timeframe direction (12h) + single-timeframe entry (1h) = optimal frequency
4) Session filter (8-20 UTC) reduces noise but don't over-filter
5) Position size 0.25 for 1h (smaller than 4h's 0.30 due to more trades)

Strategy Logic:
- 12h HMA(21): Macro trend bias (only long if price > 12h HMA, only short if <)
- 4h Choppiness(14): Regime detection (CHOP>55 = range/mean-revert, CHOP<45 = trend)
- 1h Connors RSI: Entry timing (CRSI<20 long, CRSI>80 short)
- Volume filter: Only enter if volume > 0.7x 20-bar avg (loose to ensure trades)
- Session: 8-20 UTC only (reduces Asian session noise)

Why this should work on ALL symbols:
- Loose CRSI thresholds ensure entries during volatility
- 12h bias prevents counter-trend in bear markets (critical for BTC/ETH 2022)
- Regime switch adapts to market conditions automatically
- 1h TF = enough trades (40-80/year) without fee drag (>100/year)

Position size: 0.25 (discrete, conservative for 1h)
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h12h_v1"
timeframe = "1h"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 20, Short when CRSI > 80
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    
    # RSI Streak (consecutive up/down bars)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 / (streak[i] + 1)
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 - (100.0 / (abs(streak[i]) + 1))
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # CRSI
    crsi = (rsi + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for macro bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h Choppiness for regime
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Discrete, conservative for 1h
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_4h_aligned[i]) or np.isnan(vol_avg_20[i]):
            continue
        if atr_14[i] == 0 or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === 12H MACRO BIAS ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        chop_value = chop_4h_aligned[i]
        is_ranging = chop_value > 55.0  # Range market
        is_trending = chop_value < 45.0  # Trend market
        
        # === VOLUME FILTER (LOOSE) ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === CONNORS RSI SIGNALS (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # Long signal (loose)
        crsi_overbought = crsi[i] > 75.0  # Short signal (loose)
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours
        if in_session and volume_ok:
            # --- RANGING REGIME: Connors RSI Mean Reversion ---
            if is_ranging:
                # Long: CRSI oversold + 12h bias helps
                if crsi_oversold:
                    if price_above_hma_12h or crsi_rising:
                        new_signal = POSITION_SIZE
                
                # Short: CRSI overbought + 12h bias helps
                elif crsi_overbought:
                    if price_below_hma_12h or crsi_falling:
                        new_signal = -POSITION_SIZE
            
            # --- TRENDING REGIME: Follow 12h trend on CRSI pullback ---
            elif is_trending:
                # Long: 12h bullish + CRSI pullback (not overbought)
                if price_above_hma_12h and crsi[i] < 60.0 and crsi_rising:
                    new_signal = POSITION_SIZE
                
                # Short: 12h bearish + CRSI pullback (not oversold)
                elif price_below_hma_12h and crsi[i] > 40.0 and crsi_falling:
                    new_signal = -POSITION_SIZE
            
            # --- NEUTRAL REGIME: CRSI extremes only (ensures trades) ---
            else:
                # Long: Very oversold CRSI
                if crsi[i] < 20.0:
                    new_signal = POSITION_SIZE
                # Short: Very overbought CRSI
                elif crsi[i] > 80.0:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if CRSI not at opposite extreme
            if position_side > 0 and crsi[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 30.0:
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
        
        # === EXIT ON 12H TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_12h:
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