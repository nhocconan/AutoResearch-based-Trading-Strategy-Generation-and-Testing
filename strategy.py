#!/usr/bin/env python3
"""
Experiment #021: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Based on research showing Connors RSI has 75% win rate for mean reversion
and Choppiness Index successfully switches between regime types (ETH Sharpe +0.923),
I'm combining these at 4h timeframe with 1d HMA for trend bias.

Key innovations:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive than regular RSI, catches reversals faster
   - Long when CRSI < 15, Short when CRSI > 85
2. CHOPPINESS INDEX regime filter: CHOP(14) > 61.8 = range (mean revert),
   CHOP < 38.2 = trend (breakout follow), middle = flat
3. 1d HMA for overall bias (only long if price > 1d HMA, vice versa)
4. Volume confirmation: entry volume > 1.3 * 20-bar avg (avoids fake breakouts)
5. ATR(14) trailing stoploss at 2.5x

Why 4h works:
- Targets 20-50 trades/year (fee-efficient per Rule 10)
- Proven in research for crypto perpetual futures
- Less noise than 1h, more signals than 12h

Position size: 0.25 (discrete, conservative for drawdown control)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_hma_1d_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak - consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak_abs[max(0, i-streak_period+1):i+1]
        if len(streak_window) > 0 and streak_window.max() > 0:
            streak_rsi[i] = 100.0 * streak_abs[i] / (streak_window.max() + 1e-10)
        else:
            streak_rsi[i] = 50.0
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - today's return vs last 100 days
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1]
        if len(window) > 0:
            percent_rank[i] = 100.0 * (window.iloc[:-1] < returns.iloc[i]).sum() / (len(window) - 1)
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
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

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for regime bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(hma_4h[i]):
            continue
        if np.isnan(vol_sma_20[i]) or atr_14[i] == 0 or vol_sma_20[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-3] if i >= 3 else False
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * vol_sma_20[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + price above 1d HMA (bullish bias)
            if crsi_oversold and price_above_hma_1d:
                new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price below 1d HMA (bearish bias)
            elif crsi_overbought and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Pullback Entries ---
        elif is_trending:
            # Long in uptrend: CRSI pullback + 4h HMA bullish + volume
            if crsi_oversold and hma_4h_slope_bull and price_above_hma_4h and volume_confirmed:
                new_signal = POSITION_SIZE
            
            # Short in downtrend: CRSI bounce + 4h HMA bearish + volume
            elif crsi_overbought and hma_4h_slope_bear and price_below_hma_4h and volume_confirmed:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime changes to trending bearish
        if in_position and position_side > 0:
            if is_trending and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short if regime changes to trending bullish
        if in_position and position_side < 0:
            if is_trending and hma_4h_slope_bull:
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