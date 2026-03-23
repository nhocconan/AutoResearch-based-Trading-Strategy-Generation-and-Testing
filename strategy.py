#!/usr/bin/env python3
"""
Experiment #406: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA + Donchian

Hypothesis: Combining Choppiness Index regime detection with Connors RSI (CRSI) for 
mean-reversion entries + HMA trend filter + Donchian breakout confirmation will beat 
Sharpe=0.612 on 12h timeframe.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven 75% win rate in research notes (ETH Sharpe +0.923)
   - Long: CRSI < 15, Short: CRSI > 85
2. Choppiness Index regime: >61.8 = range (favor mean revert), <38.2 = trend
3. HMA(21/50) crossover for trend direction bias
4. 1d HTF HMA for overall market bias filter
5. Donchian(20) for breakout confirmation in trending regime
6. ATR(14) trailing stoploss (2.5x asymmetric)
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
8. Relaxed entry thresholds to ensure ≥30 trades/year per symbol

Why 12h works:
- Target 20-50 trades/year = ~1-2 trades per month per symbol
- Fee drag ~1-2.5% annually (much lower than 1h/4h strategies)
- Captures major swings without whipsaw noise
- Proven in #396, #402 notes (SOL Sharpe +0.782, +0.879)

Why CRSI beats regular RSI:
- 3-period RSI reacts faster to oversold/overbought
- Streak component captures momentum exhaustion
- PercentRank normalizes across different vol regimes
- Combined gives more reliable reversal signals

Target: Sharpe > 0.612, 25-50 trades/year, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Streak RSI: RSI applied to up/down streak lengths
    PercentRank: % of past N closes that are below current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:][~np.isnan(atr_14[100:])]
    atr_median = np.median(valid_atr) if len(valid_atr) > 0 else 1.0
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (target 25-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market - favor mean reversion
        is_trending = chop[i] < 38.2  # Trend market - favor breakouts
        # Neutral zone: 38.2 <= CHOP <= 61.8 - use both signals
        
        # === HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA crossover) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === CONNORS RSI EXTREMES (relaxed for more trades) ===
        # Long: CRSI < 20 (oversold), Short: CRSI > 80 (overbought)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_long = close[i] > donchian_upper[i-1]
        donchian_short = close[i] < donchian_lower[i-1]
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple confluence paths (relaxed for more trades)
        long_bias = price_above_hma_1d or hma_bullish  # Either HTF or LTF bullish
        
        if long_bias:
            if is_choppy:
                # Mean reversion in range: CRSI oversold + RSI confirmation
                if crsi_oversold and rsi_oversold:
                    desired_signal = position_size
                elif crsi_oversold and close[i] < (donchian_lower[i] + 0.03 * close[i]):
                    # CRSI oversold near Donchian lower
                    desired_signal = position_size
            elif is_trending:
                # Trend following: HMA bullish + Donchian breakout
                if hma_bullish and donchian_long:
                    desired_signal = position_size
                elif hma_bullish and crsi_oversold:
                    # Pullback in uptrend
                    desired_signal = position_size
            else:
                # Neutral regime: use either signal
                if crsi_oversold:
                    desired_signal = position_size
                elif hma_bullish and donchian_long:
                    desired_signal = position_size
        
        # SHORT SETUP - Multiple confluence paths (relaxed for more trades)
        short_bias = price_below_hma_1d or hma_bearish  # Either HTF or LTF bearish
        
        if short_bias:
            if is_choppy:
                # Mean reversion in range: CRSI overbought + RSI confirmation
                if crsi_overbought and rsi_overbought:
                    desired_signal = -position_size
                elif crsi_overbought and close[i] > (donchian_upper[i] - 0.03 * close[i]):
                    # CRSI overbought near Donchian upper
                    desired_signal = -position_size
            elif is_trending:
                # Trend following: HMA bearish + Donchian breakdown
                if hma_bearish and donchian_short:
                    desired_signal = -position_size
                elif hma_bearish and crsi_overbought:
                    # Rally in downtrend
                    desired_signal = -position_size
            else:
                # Neutral regime: use either signal
                if crsi_overbought:
                    desired_signal = -position_size
                elif hma_bearish and donchian_short:
                    desired_signal = -position_size
        
        # === STOPLOSS CHECK (Asymmetric: 2.5x for longs, 2.0x for shorts) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXIT (extreme reached - take profit) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            # Long exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            # Short exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === RSI EXIT (extreme reached) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1d and hma_bearish:
            # Both HTF and LTF turned bearish
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and hma_bullish:
            # Both HTF and LTF turned bullish
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias:
                desired_signal = position_size
            elif position_side < 0 and short_bias:
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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