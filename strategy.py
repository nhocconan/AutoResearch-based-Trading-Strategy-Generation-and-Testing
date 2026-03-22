#!/usr/bin/env python3
"""
Experiment #618: 30m Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion with Connors RSI

Hypothesis: 30m strategies fail due to (1) too many trades → fee drag, or (2) too few → 0 trades.
This strategy balances both by using:
- 1d Choppiness Index for regime detection (proven on 1d in current best Sharpe=0.520)
- 4h HMA(21) for primary trend bias (smooth, less whipsaw than EMA)
- 30m Connors RSI for precise entry timing (more sensitive than standard RSI)
- Volume confirmation (0.7x avg) to filter noise without over-filtering
- NO session filter (caused 0 trades in #608, #615)

Key learnings from 546 failed strategies:
1. Choppiness works on 1d/1w, NOT on 4h/12h (all chop+4h strategies failed)
2. Connors RSI better than standard RSI for mean reversion entries
3. Volume filter must be relaxed (0.7x not 1.5x) to ensure trades
4. 30m needs HTF trend filter to avoid counter-trend trades

Position sizing: 0.25 (conservative for 30m, per Rule 4 max 0.40)
Target: 40-80 trades/year (per Rule 10 for 30m)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_crsi_hma_4h1d_v1"
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
    """Calculate RSI using Wilder's smoothing."""
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
    More sensitive than standard RSI, better for mean reversion entries.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period:i+1]
        gains = np.sum(streak_window[streak_window > 0])
        losses = np.abs(np.sum(streak_window[streak_window < 0]))
        if losses == 0:
            streak_rsi[i] = 100.0
        else:
            rs = gains / losses
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (len(window) - 1) if len(window) > 1 else 0.5
        percent_rank[i] = rank * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hma = (2.0 * wma_half - wma_full).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 30m indicators
    crsi_30m = calculate_connors_rsi(close, 3, 2, 100)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Bollinger Bands (20, 2.0)
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    # Volume SMA (20)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(crsi_30m[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(vol_sma[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D REGIME (Choppiness) ===
        is_trend_regime = chop_1d_aligned[i] < 45.0
        is_range_regime = chop_1d_aligned[i] > 55.0
        
        # === 4H TREND BIAS (HMA slope) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (relaxed to 0.7x) ===
        volume_ok = volume[i] > 0.7 * vol_sma[i]
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 4h trend on CRSI pullbacks ---
        if is_trend_regime:
            # LONG: 4h bull trend + CRSI oversold (20-40) + volume
            if hma_4h_slope_bull and price_above_hma_4h and volume_ok:
                if 20.0 <= crsi_30m[i] <= 40.0:
                    new_signal = POSITION_SIZE
            
            # SHORT: 4h bear trend + CRSI overbought (60-80) + volume
            elif hma_4h_slope_bear and price_below_hma_4h and volume_ok:
                if 60.0 <= crsi_30m[i] <= 80.0:
                    new_signal = -POSITION_SIZE
        
        # --- RANGE REGIME: Mean reversion at extremes ---
        elif is_range_regime:
            # LONG: CRSI < 25 (deep oversold) + price near BB lower + volume
            if crsi_30m[i] < 25.0 and close[i] <= bb_lower[i] and volume_ok:
                new_signal = POSITION_SIZE
            
            # SHORT: CRSI > 75 (deep overbought) + price near BB upper + volume
            elif crsi_30m[i] > 75.0 and close[i] >= bb_upper[i] and volume_ok:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals