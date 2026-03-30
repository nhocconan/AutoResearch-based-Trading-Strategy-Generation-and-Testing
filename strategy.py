#!/usr/bin/env python3
"""
Experiment #028: 4h CRSI + Volume + 1d SMA200 Trend

HYPOTHESIS: CRSI (Connors RSI) combines three momentum indicators to identify 
high-probability extremes with 75%+ win rate. Combined with 1d SMA200 for trend
direction and volume confirmation, this captures mean-reversion setups at key levels.

WHY IT WORKS IN BULL AND BEAR: Long when CRSI<15 + price>SMA200 (bull dips).
Short when CRSI>85 + price<SMA200 (bear rallies). Symmetric logic works both directions.

TARGET: 75-200 total trades over 4 years. Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_vol_sma200_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(prices, rsi_len=3, streak_len=2, rank_len=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    close = prices['close'].values if hasattr(prices, 'close') else prices
    n = len(close)
    
    # Component 1: RSI(3) with min_periods
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    
    up = np.maximum(delta, 0)
    down = np.maximum(-delta, 0)
    
    ema_up = pd.Series(up).ewm(span=rsi_len, min_periods=rsi_len, adjust=False).mean().values
    ema_down = pd.Series(down).ewm(span=rsi_len, min_periods=rsi_len, adjust=False).mean().values
    
    rs = np.divide(ema_up, np.maximum(ema_down, 1e-10))
    rsi3 = 100 - (100 / (1 + rs))
    
    # Component 2: RSI Streak (consecutive up/down bars)
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + delta[i] if streak[i-1] >= 0 else delta[i]
        elif delta[i] < 0:
            streak[i] = streak[i-1] + delta[i] if streak[i-1] <= 0 else delta[i]
    
    streak_up = np.maximum(streak, 0)
    streak_down = np.maximum(-streak, 0)
    
    ema_su = pd.Series(streak_up).ewm(span=streak_len, min_periods=streak_len, adjust=False).mean().values
    ema_sd = pd.Series(streak_down).ewm(span=streak_len, min_periods=streak_len, adjust=False).mean().values
    
    rs_streak = np.divide(ema_su, np.maximum(ema_sd, 1e-10))
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Component 3: PercentRank(100) with min_periods
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_len - 1, n):
        window = close[i - rank_len + 1:i + 1]
        rank = (window < close[i]).sum()
        percent_rank[i] = (rank / rank_len) * 100
    
    # CRSI = average of three components
    crsi = (rsi3 + rsi_streak + percent_rank) / 3
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(prices, rsi_len=3, streak_len=2, rank_len=100)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + 50 buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if SMA200 not ready
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if CRSI not ready
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # === Trend direction ===
        price_above_sma = close[i] > sma_1d_aligned[i]
        price_below_sma = close[i] < sma_1d_aligned[i]
        
        # === Volume confirmation ===
        vol_confirm = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        bars_held = i - entry_bar
        
        # === EXIT LOGIC ===
        if position != 0:
            # Time-based hold minimum: 4 bars (16 hours)
            if bars_held >= 4:
                # CRSI mean reversion exit
                if position > 0 and crsi[i] > 50:
                    desired_signal = 0.0
                elif position < 0 and crsi[i] < 50:
                    desired_signal = 0.0
            
            # Stop loss: 2.5 ATR
            if position > 0 and low[i] < entry_price - 2.5 * entry_atr:
                desired_signal = 0.0
            elif position < 0 and high[i] > entry_price + 2.5 * entry_atr:
                desired_signal = 0.0
            
            # Take profit: CRSI overbought/oversold reversal
            if position > 0 and crsi[i] > 70:
                desired_signal = 0.0
            elif position < 0 and crsi[i] < 30:
                desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if position == 0:
            # LONG: CRSI oversold + price above SMA200 + volume confirm
            if crsi[i] < 15 and price_above_sma and vol_confirm:
                desired_signal = SIZE
            
            # SHORT: CRSI overbought + price below SMA200 + volume confirm
            elif crsi[i] > 85 and price_below_sma and vol_confirm:
                desired_signal = -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if desired_signal != position:
                # New entry or flip
                position = np.sign(desired_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            position = 0
        
        signals[i] = desired_signal
    
    return signals