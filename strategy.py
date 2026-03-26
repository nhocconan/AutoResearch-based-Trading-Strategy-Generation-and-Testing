#!/usr/bin/env python3
"""
Experiment #021: 12h Volatility Expansion + RSI Extremes + 1d Trend

HYPOTHESIS: Markets cycle between low-volatility squeeze and high-volatility 
expansion phases. Entering on volatility expansion (ATR(14) > 1.8× ATR(100)) 
with RSI(7) at extremes captures the start of big moves. The 1d SMA(50) keeps 
direction aligned with the larger trend, avoiding countertrend trades in 
strong trends. This regime-adaptive approach works in both bull (buy RSI dips 
with 1d uptrend) and bear (short RSI spikes with 1d downtrend).

WHY 12H: Slower than 4h = fewer trades = less fee drag. ATR(100) on 12h = 
1200h lookback = 50 days = good regime filter. RSI(7) on 12h = equivalent to 
RSI(28) on 1h = smooth, less noise.

EXPECTED TRADES: 60-120 total over 4 years (15-30/year). Volume filter adds 
confirmation to avoid false breakouts.

TARGET: Sharpe > 0.5 on train, trades >= 50 total.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volatility_expansion_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper handling"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI calculation"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for trend filter
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_100 = calculate_atr(high, low, close, period=100)
    
    rsi_7 = calculate_rsi(close, period=7)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    entry_bar = 0
    
    warmup = max(100, 100)  # ATR(100) needs ~100 bars
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_100[i]) or atr_100[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK: Volatility Expansion ===
        atr_ratio = atr_14[i] / atr_100[i]
        vol_expansion = atr_ratio > 1.8
        
        # === TREND FILTER (1d SMA) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === MOMENTUM (RSI 7) ===
        rsi_val = rsi_7[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Volatility expansion + RSI oversold + bullish 1d trend + volume confirm
            if vol_expansion and rsi_val < 35 and price_above_1d_sma and vol_confirm:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Volatility expansion + RSI overbought + bearish 1d trend + volume confirm
            if vol_expansion and rsi_val > 65 and not price_above_1d_sma and vol_confirm:
                desired_signal = -SIZE
        
        else:
            # === STOPLOSS CHECK ===
            stop_triggered = False
            
            if position_side > 0:
                if low[i] < stop_price:
                    stop_triggered = True
            elif position_side < 0:
                if high[i] > stop_price:
                    stop_triggered = True
            
            if stop_triggered:
                desired_signal = 0.0
            else:
                # === TREND LOSS CHECK ===
                # If trend flips against position, exit
                if position_side > 0 and not price_above_1d_sma:
                    # 1d trend turned bearish - exit long
                    desired_signal = 0.0
                elif position_side < 0 and price_above_1d_sma:
                    # 1d trend turned bullish - exit short
                    desired_signal = 0.0
                else:
                    # === RSI NEUTRAL EXIT ===
                    # RSI returned to neutral - take profit
                    if position_side > 0 and rsi_val > 55:
                        desired_signal = 0.0
                    elif position_side < 0 and rsi_val < 45:
                        desired_signal = 0.0
                    else:
                        # Maintain position
                        desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                # Stop: 2.5 ATR from entry
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals