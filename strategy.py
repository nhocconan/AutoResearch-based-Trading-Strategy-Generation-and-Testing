#!/usr/bin/env python3
"""
Experiment #022: 1d Donchian Breakout + Volume + 1w Trend Filter

HYPOTHESIS: Simple but robust 1d strategy that captures major trend shifts:
1. Donchian(20) breakout - captures trend momentum
2. Volume spike confirmation - ensures institutional participation
3. 1w SMA50 filter - confirms trend direction on higher timeframe
4. ATR stoploss - risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Breakout above 20d high + volume spike + price > SMA50 = strong long
- Bear: Breakout below 20d low + volume spike + price < SMA50 = strong short
- Simple breakout logic works in all market conditions

KEY INSIGHT from DB: Simple strategies with tight but not too tight conditions
generate 75-150 train trades and generalize well to test.

TARGET: 50-150 total over 4 years (12.5-37/year on 1d).
Primary = 1d, HTF = 1w, leverage = 1.0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper (highest high) and lower (lowest low)"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w SMA50 for trend direction ===
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # SMA50 for local trend
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume ratio
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
    
    warmup = 100  # Donchian needs 20, SMA50 needs 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend
        htf_bull = not np.isnan(sma_1w_aligned[i]) and close[i] > sma_1w_aligned[i]
        htf_bear = not np.isnan(sma_1w_aligned[i]) and close[i] < sma_1w_aligned[i]
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout above 20d high + volume spike + price > SMA50 + HTF bull
            long_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
            price_above_sma = close[i] > sma_50[i]
            
            if long_breakout and vol_spike and price_above_sma and htf_bull:
                desired_signal = SIZE
            
            # SHORT: Breakout below 20d low + volume spike + price < SMA50 + HTF bear
            short_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
            price_below_sma = close[i] < sma_50[i]
            
            if short_breakout and vol_spike and price_below_sma and htf_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Also exit if price falls below SMA50 (trend weakening)
                if close[i] < sma_50[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Also exit if price rises above SMA50 (trend weakening)
                if close[i] > sma_50[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals