#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + 1d Trend + Volume Confirmation

HYPOTHESIS: 12h timeframe with 1d HTF trend filter for robust entries:
1. 12h Donchian(20) breakout - captures multi-day momentum shifts
2. 1d SMA(50) for trend direction - bull when price > SMA50, bear when < SMA50
3. Volume spike confirmation - validates breakout strength
4. ATR stoploss - risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Breakout above Donchian high + price > SMA50(1d) = trend continuation long
- Bear: Breakout below Donchian low + price < SMA50(1d) = trend continuation short
- 12h timeframe naturally limits trades (targeting 12-37/year = 50-150 over 4 years)
- Donchian breakout is a proven structure in DB (test Sharpe 1.1-1.5)

SIMPLIFIED from previous complex strategies (Ichimoku/Alligator had too many conditions).
KEY INSIGHT: Fewer conditions = fewer trades = less fee drag = better generalization.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_trend_vol_v5"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d SMA(50) for trend direction ===
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    
    # HTF trend: price above SMA50 = bull, below = bear
    htf_price_1d = df_1d['close'].values
    htf_bullish = htf_price_1d > sma_50_1d
    htf_bearish = htf_price_1d < sma_50_1d
    
    # Align HTF to primary TF (auto shift by 1 to avoid look-ahead)
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20) - 20 bars of 12h = ~10 trading days
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume ratio (current vs 20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Donchian needs 20, ATR needs 14
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        # HTF alignment check
        htf_bull = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # Donchian breakout signals
        bull_breakout = high[i] > donch_upper[i]  # Break above 20-bar high
        bear_breakout = low[i] < donch_lower[i]   # Break below 20-bar low
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bull breakout + HTF bull + volume spike
            if bull_breakout and vol_spike and htf_bull:
                desired_signal = SIZE
            
            # SHORT: Bear breakout + HTF bear + volume spike
            elif bear_breakout and vol_spike and htf_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
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