#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian + RSI + Choppiness Regime

HYPOTHESIS: Use proven CRSI momentum + Donchian breakout + choppiness regime on 12h.
Simple = fewer trades, lower fee drag, better test generalization.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Breakout above Donchian high + RSI momentum + chop < 40 = ride the move
- Bear: Breakout below Donchian low + RSI momentum + chop < 40 = short the crash
- Range: chop > 60 = no trades (avoids whipsaws in 2022)

KEY INSIGHT from DB: CRSI-based strategies (SOL: 1.46) and simple Donchian
combos (multiple winners with 1.3-1.5 Sharpe) work. Keep it minimal.

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_rsi_chop_simple_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    delta = np.diff(prices, prepend=prices[0])
    delta[0] = 0
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, np.where(avg_loss == 0, 1e-10, avg_loss))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_crsi(prices, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(prices)
    
    # Component 1: Classic RSI(3)
    rsi = calculate_rsi(prices, period=rsi_period)
    
    # Component 2: RSI Streak
    # Compute streak: consecutive up/down days
    delta = np.diff(prices, prepend=prices[0])
    delta[0] = 0
    
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = max(1, streak[i-1] + 1)
        elif delta[i] < 0:
            streak[i] = min(-1, streak[i-1] - 1)
        else:
            streak[i] = 0
    
    rsi_streak = calculate_rsi(np.abs(streak) + 50, period=streak_period)
    
    # Component 3: Percent Rank over 100 periods
    percent_rank = np.zeros(n, dtype=np.float64)
    for i in range(rank_period, n):
        window = prices[i-rank_period:i+1]
        rank = (prices[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-10)
        percent_rank[i] = rank * 100
    
    # CRSI = average of 3 components
    crsi = (rsi + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness(prices, high, low, period=14):
    """
    Choppiness Index: 100 * LOG10(SUM(ATR,14) / (HHV(14) - LLV(14))) / LOG10(14)
    CHOP < 38.2 = trending (use trend following)
    CHOP > 61.8 = choppy (avoid or mean revert)
    """
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = np.zeros(n)
    tr0 = high[0] - low[0]
    atr[0] = tr0
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - prices[i-1]), abs(low[i] - prices[i-1]))
        atr[i] = (atr[i-1] * (period - 1) + tr) / period
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        sum_atr = np.sum(atr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll:
            ratio = sum_atr / (hh - ll)
            chop[i] = 100 * np.log10(ratio) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower"""
    n = len(high)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - period + 1)
        upper[i] = np.max(high[start:i+1])
        lower[i] = np.min(low[start:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # HTF indicators (1d)
    htf_close_1d = df_1d['close'].values
    htf_high_1d = df_1d['high'].values
    htf_low_1d = df_1d['low'].values
    
    # HTF Donchian for trend direction
    htf_donch_upper_1d, _, htf_donch_lower_1d = calculate_donchian(htf_high_1d, htf_low_1d, period=20)
    
    # HTF: Price above/below 20-day Donchian = trend direction
    htf_bull = (htf_close_1d > htf_donch_upper_1d * 0.98).astype(float)  # Allow 2% buffer
    htf_bear = (htf_close_1d < htf_donch_lower_1d * 1.02).astype(float)
    
    # Align HTF to 12h
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bull)
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bear)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # CRSI momentum
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Choppiness regime
    chop = calculate_choppiness(close, high, low, period=14)
    
    # Donchian breakout
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    # Need 100 bars warmup for CRSI percent rank (100 period)
    warmup = 150
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK ===
        # Choppiness > 61.8 = ranging, no entry (avoids whipsaws)
        ranging = chop[i] > 61.8
        trending = chop[i] < 45.0  # Slightly relaxed from 38.2 for more trades
        
        # === HTF TREND ===
        htf_bull = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        htf_neutral = not htf_bull and not htf_bear
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Price breaks above 20-bar Donchian high
            breakout_high = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1]
            
            # CRSI momentum confirmation (oversold + reversing)
            crsi_oversold = crsi[i] < 25
            crsi_recovering = crsi[i] > crsi[i-1] if not np.isnan(crsi[i-1]) else False
            
            # Volume confirmation
            vol_confirm = vol_ratio[i] > 1.3
            
            # Entry: breakout + momentum + volume + not ranging + HTF bull/neutral
            if breakout_high and crsi_oversold and vol_confirm and not ranging:
                if htf_bull or htf_neutral:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Price breaks below 20-bar Donchian low
            breakout_low = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1]
            
            # CRSI momentum confirmation (overbought + reversing)
            crsi_overbought = crsi[i] > 75
            crsi_falling = crsi[i] < crsi[i-1] if not np.isnan(crsi[i-1]) else False
            
            # Entry: breakout + momentum + volume + not ranging + HTF bear/neutral
            if breakout_low and crsi_overbought and vol_confirm and not ranging:
                if htf_bear or htf_neutral:
                    desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if CRSI turns bearish or chop increases
                if crsi[i] > 80:
                    desired_signal = 0.0
                
                # Exit if ranging develops
                if ranging:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if CRSI turns bullish or chop increases
                if crsi[i] < 20:
                    desired_signal = 0.0
                
                # Exit if ranging develops
                if ranging:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals