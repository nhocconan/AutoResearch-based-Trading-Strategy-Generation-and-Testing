#!/usr/bin/env python3
"""
Experiment #005: 4h Primary + 1d HTF — Connors RSI + Choppiness Dual Regime

Hypothesis: Combining Connors RSI (proven 75% win rate on reversals) with 
Choppiness Index regime detection will outperform pure vol-spike strategies.

Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive to short-term reversals than standard RSI
2. Choppiness Index determines regime:
   - CHOP > 61.8: Range-bound → mean reversion entries
   - CHOP < 38.2: Trending → momentum breakout entries
3. Dual-mode logic adapts to market conditions
4. 1d HMA for HTF trend bias (only trade with HTF trend in trending regime)

Why this might work better than #004:
- CRSI catches reversals earlier than BB %B
- CHOP filter prevents mean-reversion in strong trends
- Fewer false signals in trending markets
- Research shows CRSI + regime filter = 0.8+ Sharpe on BTC/ETH

Entry conditions (loose enough for trades):
- Long: CRSI < 20 OR (CHOP > 61.8 + BB %B < 0.2)
- Short: CRSI > 80 OR (CHOP > 61.8 + BB %B > 0.8)
- In trending regime: require 1d HMA alignment

Position size: 0.28 (conservative for 4h)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_dual_regime_1d_v1"
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

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over last 100 bars
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - measure consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(2, n):
        window = streak[max(0, i-2):i+1]
        pos_count = np.sum(window > 0)
        total = len(window)
        if total > 0:
            streak_rsi[i] = 100.0 * pos_count / total
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank(100) - percentile of current change in last 100 bars
    pct_rank = np.zeros(n)
    for i in range(100, n):
        changes = np.diff(close[max(0, i-100):i+1])
        if len(changes) > 0:
            current_change = close[i] - close[i-1]
            pct_rank[i] = 100.0 * np.sum(changes <= current_change) / len(changes)
        else:
            pct_rank[i] = 50.0
    
    # Fill early values
    pct_rank[:100] = 50.0
    
    # CRSI
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range-bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr_vals = calculate_atr(high, low, close, period=period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B (position within bands)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    return upper.values, lower.values, pct_b.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start after indicators warm up (CRSI needs 100 + RSI needs warmup)
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(bb_pct_b[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === REGIME DETECTION ===
        choppy_regime = chop[i] > 61.8
        trending_regime = chop[i] < 38.2
        
        # === 1D TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === LONG-TERM BIAS ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === DUAL REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        if choppy_regime:
            # MEAN REVERSION MODE - trade both directions at extremes
            # Long: CRSI extreme oversold OR BB lower band
            if crsi[i] < 20 or (bb_pct_b[i] < 0.15 and crsi[i] < 40):
                new_signal = POSITION_SIZE
            
            # Short: CRSI extreme overbought OR BB upper band
            elif crsi[i] > 80 or (bb_pct_b[i] > 0.85 and crsi[i] > 60):
                new_signal = -POSITION_SIZE
        
        elif trending_regime:
            # TREND FOLLOWING MODE - only trade with 1d trend direction
            if hma_1d_slope_bull and price_above_hma_1d:
                # Long on pullback in uptrend
                if crsi[i] < 40 or bb_pct_b[i] < 0.4:
                    new_signal = POSITION_SIZE
            
            elif hma_1d_slope_bear and price_below_hma_1d:
                # Short on pullback in downtrend
                if crsi[i] > 60 or bb_pct_b[i] > 0.6:
                    new_signal = -POSITION_SIZE
        
        else:
            # NEUTRAL REGIME (38.2 <= CHOP <= 61.8)
            # Use CRSI extremes only, require SMA200 alignment
            if crsi[i] < 15 and price_above_sma200:
                new_signal = POSITION_SIZE
            elif crsi[i] > 85 and price_below_sma200:
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