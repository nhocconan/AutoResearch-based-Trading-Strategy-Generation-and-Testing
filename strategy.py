#!/usr/bin/env python3
"""
Experiment #032: 12h KAMA-Choppiness-Connors with 1d Trend Filter

Hypothesis: Previous HMA-Donchian strategies failed due to whipsaw in ranging markets
(2022-2025 has been mostly range/bear). KAMA adapts to market efficiency (ER),
slowing in chop and speeding in trends. Combined with Choppiness Index regime filter
and Connors RSI for entry timing, this should reduce false signals.

Key components:
1. KAMA(10) - Adaptive MA that adjusts to market efficiency ratio
2. Choppiness Index(14) - Regime filter: >61.8 = range, <38.2 = trend
3. Connors RSI - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. 1d KAMA(21) - Major trend bias filter
5. Asymmetric sizing: larger in trending regime, smaller in ranging

Why this should work:
- KAMA reduces whipsaw in chop (ER low = slow adaptation)
- Choppiness filter avoids trend entries in range markets
- Connors RSI catches pullbacks within trend (75% win rate in research)
- 12h TF = 20-50 trades/year target (fee drag manageable)
- 1d filter ensures we trade with major trend

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 (smaller in chop regime)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_connors_1d_trend_atr_v1"
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

def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/31):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency ratio.
    ER near 1 = trending (fast SC), ER near 0 = choppy (slow SC)
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    signal = np.abs(close - np.roll(close, er_period))
    signal[:er_period] = np.nan
    
    noise = np.zeros(n)
    for i in range(er_period, n):
        noise[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    noise[:er_period] = np.nan
    
    er = signal / np.where(noise > 0, noise, 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = range-bound, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_short = 100 - (100 / (1 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100 * min(streak[i], streak_period) / streak_period
        else:
            streak_rsi[i] = 100 * (1 + max(streak[i], -streak_period) / streak_period)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_short.values + streak_rsi + percent_rank) / 3
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    kama_1d_21 = calculate_kama(df_1d['close'].values, er_period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_10 = calculate_kama(close, er_period=10)
    kama_12h_30 = calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/31)
    # Recalculate kama_12h_30 with slower parameters
    kama_12h_30 = calculate_kama(close, er_period=10, fast_sc=2/21, slow_sc=2/41)
    
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_TREND = 0.30  # Larger in trending regime
    BASE_SIZE_CHOP = 0.20   # Smaller in choppy regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_30[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > kama_1d_21_aligned[i]
        daily_bearish = close[i] < kama_1d_21_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = kama_12h_10[i] > kama_12h_30[i]
        kama_bearish = kama_12h_10[i] < kama_12h_30[i]
        
        # === KAMA SLOPE ===
        kama_slope_long = kama_12h_10[i] > kama_12h_10[i-3] if i > 3 else False
        kama_slope_short = kama_12h_10[i] < kama_12h_10[i-3] if i > 3 else False
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 55  # Range-bound market
        chop_trend = chop_14[i] < 45  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 25  # Entry long
        crsi_overbought = crsi[i] > 75  # Entry short
        
        # === REGIME-ADAPTIVE POSITION SIZING ===
        if chop_trend:
            current_size = BASE_SIZE_TREND
        else:
            current_size = BASE_SIZE_CHOP
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Trend + CRSI pullback (regime-aware)
        if kama_bullish and daily_bullish:
            if crsi_oversold:
                # Strong signal: trend + pullback
                new_signal = current_size
            elif chop_range and crsi[i] < 35:
                # Weaker signal in range: deeper oversold needed
                new_signal = current_size * 0.7
        
        # SHORT ENTRY: Trend + CRSI pullback (regime-aware)
        if kama_bearish and daily_bearish:
            if crsi_overbought:
                # Strong signal: trend + pullback
                new_signal = -current_size
            elif chop_range and crsi[i] > 65:
                # Weaker signal in range: deeper overbought needed
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~40 days on 12h), allow weaker entries
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if kama_bullish and crsi[i] < 40:
                new_signal = current_size * 0.6
            elif kama_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h KAMA turns bearish
            if position_side > 0 and kama_bearish:
                trend_reversal = True
            # Exit short if 12h KAMA turns bullish
            if position_side < 0 and kama_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT (take profit) ===
        take_profit = False
        if in_position and position_side != 0:
            # Exit long when CRSI overbought
            if position_side > 0 and crsi[i] > 85:
                take_profit = True
            # Exit short when CRSI oversold
            if position_side < 0 and crsi[i] < 15:
                take_profit = True
        
        # Apply stoploss, trend reversal, or take profit
        if stoploss_triggered or trend_reversal or take_profit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # else: same direction, maintain position (no signal change = no fee)
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals