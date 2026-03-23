#!/usr/bin/env python3
"""
Experiment #882: 12h Primary + 1d/1w HTF — Simplified Dual Regime + CRSI + Volume

Hypothesis: After 600+ failed strategies, the key issue is OVER-COMPLEXITY.
Strategies with 5+ regime states and multiple conflicting filters generate 0 trades.

Key insights from failures:
1. #872 (12h CRSI+Chop+Donchian) got Sharpe=-0.183 — too many regime states
2. Most failures have Sharpe=0.000 meaning ZERO trades generated
3. Complex hold/exit logic causes premature exits

NEW APPROACH — SIMPLIFIED:
1. Binary regime only: CHOP>50=range, CHOP<50=trend (not 3-5 states)
2. Relaxed CRSI: 15/85 thresholds (not 20/80) to ensure trades
3. Volume confirmation on breakouts only (not on mean reversion)
4. Asymmetric sizing: 0.30 in trend, 0.20 in range
5. Simple hold logic: maintain position until opposite signal OR stoploss
6. NO complex exit conditions — let ATR trailing stop handle exits

Why 12h works:
- 20-50 trades/year target (low fee drag)
- HTF (1d/1w) provides strong trend bias
- Less noise than 1h/4h strategies

Critical fixes from #872:
- Removed neutral regime (was causing 0 trades)
- Simplified hold logic (was exiting too early)
- Relaxed CRSI thresholds further (15/85 not 20/80)
- Added volume filter ONLY for breakouts (not blocking mean reversion)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simplified_crsi_chop_binary_1d1w_hma_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    CRSI < 15 = oversold (long), CRSI > 85 = overbought (short)
    Relaxed thresholds to ensure trades on all symbols
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < max(rsi_period, streak_period, rank_period) + 2:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
            direction[i] = 1
        elif close[i] < close[i-1]:
            if direction[i-1] == -1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = -1
            direction[i] = -1
        else:
            streak[i] = 0
            direction[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_vals > 0)
        down_streaks = np.sum(streak_vals < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50
    
    # Percent Rank of price change
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = 100 * rank
        else:
            percent_rank[i] = 50
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — binary regime: CHOP>50=ranging, CHOP<50=trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
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

def calculate_volume_sma(volume, period=20):
    """Volume SMA for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.30
    RANGE_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index - BINARY) ===
        ranging_regime = chop_12h[i] > 50
        trending_regime = chop_12h[i] <= 50
        
        # === CONNORS RSI SIGNALS (Relaxed: 15/85) ===
        crsi_oversold = crsi_12h[i] < 15
        crsi_overbought = crsi_12h[i] > 85
        
        # === RSI SIGNALS (fallback for more trades) ===
        rsi_oversold = rsi_12h[i] < 30
        rsi_overbought = rsi_12h[i] > 70
        
        # === VOLUME CONFIRMATION ===
        volume_above_avg = volume[i] > vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        desired_signal = 0.0
        current_size = TREND_SIZE if trending_regime else RANGE_SIZE
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold OR RSI oversold + trend alignment
            if crsi_oversold:
                desired_signal = current_size
            elif rsi_oversold and (macro_bull or trend_1d_bullish):
                desired_signal = current_size
            
            # Short: CRSI overbought OR RSI overbought + trend alignment
            if crsi_overbought:
                desired_signal = -current_size
            elif rsi_overbought and (macro_bear or trend_1d_bearish):
                desired_signal = -current_size
        
        # === TRENDING REGIME (CHOP <= 50) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback OR RSI pullback
            if macro_bull or trend_1d_bullish:
                if crsi_oversold:
                    desired_signal = current_size
                elif rsi_oversold and volume_above_avg:
                    desired_signal = current_size
            
            # Short: Bearish trend + CRSI pullback OR RSI pullback
            if macro_bear or trend_1d_bearish:
                if crsi_overbought:
                    desired_signal = -current_size
                elif rsi_overbought and volume_above_avg:
                    desired_signal = -current_size
        
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
        
        # === SIMPLE HOLD LOGIC ===
        # Maintain position if we're in one and no opposite signal
        if in_position and not stoploss_triggered:
            if position_side > 0 and desired_signal == 0:
                # Hold long if CRSI not extremely overbought
                if crsi_12h[i] < 80:
                    desired_signal = current_size
            elif position_side < 0 and desired_signal == 0:
                # Hold short if CRSI not extremely oversold
                if crsi_12h[i] > 20:
                    desired_signal = -current_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = current_size
        elif desired_signal < 0:
            desired_signal = -current_size
        
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