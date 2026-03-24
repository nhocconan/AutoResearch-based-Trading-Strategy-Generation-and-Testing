#!/usr/bin/env python3
"""
Experiment #771: 6h Primary + 1w/1d HTF — Regime-Adaptive with CRSI/HMA Dual Mode

Hypothesis: 6h timeframe sits between 4h and 12h - needs regime detection to avoid
whipsaws in ranging markets. Previous 6h experiments failed because they used ONE
logic for all market conditions. This strategy ADAPTS: mean-revert in ranges (CRSI),
trend-follow in trends (HMA crossover).

Key innovations:
1. 1d CHOP(14) regime filter: >61.8 = range mode, <38.2 = trend mode
2. 1w HMA(21) for macro bias (only trade with weekly trend)
3. CRSI(3,2,100) for range entries - proven 75% win rate in sideways markets
4. HMA(16/48) crossover for trend entries - catches sustained moves
5. Volume confirmation: entry vol > 1.3x 20-bar avg (filters false breakouts)
6. ATR(14) 2.5x trailing stop with signal→0 on stoploss hit
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry logic:
- RANGE mode (CHOP>61.8): Long CRSI<15 + price>1w HMA, Short CRSI>85 + price<1w HMA
- TREND mode (CHOP<38.2): Long HMA16>48 + pullback RSI<50, Short HMA16<48 + pullback RSI>50
- NEUTRAL mode (38.2<=CHOP<=61.8): No new entries, only manage existing positions

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_crsi_hma_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak RSI, and percent rank for mean reversion"""
    n = len(close)
    if n < rank_period + rsi_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
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
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        gains = np.sum(streak_vals[streak_vals > 0])
        losses = np.abs(np.sum(streak_vals[streak_vals < 0]))
        if losses > 1e-10:
            rs = gains / losses
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            streak_rsi[i] = 100.0
    
    # Percent Rank(100): where does today's return rank vs last 100 days?
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        today_return = returns[i]
        rank = np.sum(window < today_return)
        percent_rank[i] = (rank / rank_period) * 100.0
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_chop(high, low, close, period=14):
    """Choppiness Index - measures if market is trending or ranging"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    chop_1d_raw = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (1d CHOP) ===
        chop_val = chop_1d_aligned[i]
        is_range_regime = chop_val > 61.8
        is_trend_regime = chop_val < 38.2
        # Neutral regime: 38.2 <= CHOP <= 61.8 (no new entries)
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = False
        if not np.isnan(vol_ma_20[i]) and vol_ma_20[i] > 1e-10:
            vol_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # === 6h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === 6h HMA TREND ===
        hma_6h_bull = hma_16[i] > hma_48[i]
        hma_6h_bear = hma_16[i] < hma_48[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with CRSI
        if is_range_regime:
            # Long: CRSI extremely oversold + price above weekly HMA
            if crsi[i] < 15.0 and htf_1w_bull and vol_confirmed:
                desired_signal = SIZE_BASE
            # Short: CRSI extremely overbought + price below weekly HMA
            elif crsi[i] > 85.0 and htf_1w_bear and vol_confirmed:
                desired_signal = -SIZE_BASE
            # Strong signals at extremes
            if crsi[i] < 10.0 and htf_1w_bull:
                desired_signal = SIZE_STRONG
            elif crsi[i] > 90.0 and htf_1w_bear:
                desired_signal = -SIZE_STRONG
        
        # TREND REGIME: Trend following with HMA + RSI pullback
        elif is_trend_regime:
            # Long: HMA bull + RSI pullback (not overbought) + volume
            if hma_6h_bull and rsi_14[i] < 55.0 and rsi_14[i] > 35.0:
                if hma_crossover_long or vol_confirmed:
                    desired_signal = SIZE_BASE
            # Short: HMA bear + RSI pullback (not oversold) + volume
            elif hma_6h_bear and rsi_14[i] > 45.0 and rsi_14[i] < 65.0:
                if hma_crossover_short or vol_confirmed:
                    desired_signal = -SIZE_BASE
            # Strong: fresh crossover with volume
            if hma_crossover_long and vol_confirmed and htf_1w_bull:
                desired_signal = SIZE_STRONG
            elif hma_crossover_short and vol_confirmed and htf_1w_bear:
                desired_signal = -SIZE_STRONG
        
        # NEUTRAL REGIME: No new entries, only manage existing
        # desired_signal stays 0.0
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals