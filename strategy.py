#!/usr/bin/env python3
"""
Experiment #025: 12h RSI Extreme + 1d EMA Trend + Volume (12h)

HYPOTHESIS: RSI(7) extremes on 12h catch mean-reversion setups within 
the major trend. 1d EMA50 provides directional bias. Volume confirms momentum.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: RSI(7) < 25 (oversold) + price > 1d EMA50 + volume spike = strong long
- Bear: RSI(7) > 75 (overbought) + price < 1d EMA50 + volume spike = strong short
- Range: RSI extremes still work as band bounces

EXPECTED TRADES: 75-150 total over 4 years (19-37/year per symbol)
- RSI(7) < 25 happens roughly 8-12x/year per symbol in crypto
- 1d EMA50 filter reduces by ~20% 
- Volume spike (1.5x) reduces by ~25%
- Final: ~50-90 trades = within target range
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_ema50_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(prices, period=14):
    """RSI indicator"""
    close = prices if isinstance(prices, np.ndarray) else prices.values
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_ema(prices, period):
    """EMA indicator"""
    return pd.Series(prices).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema50_1d = calculate_ema(df_1d['close'].values, period=50)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # RSI(7) for momentum
    rsi_7 = calculate_rsi(close, period=7)
    
    # Volume average (20 bars = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Enough for RSI7, ATR14, EMA50(1d)
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: 1d EMA50 ===
        price_vs_ema = close[i] - ema50_aligned[i]
        pct_above_ema = price_vs_ema / ema50_aligned[i] * 100
        
        bull_trend = pct_above_ema > 0  # Price above 1d EMA50
        bear_trend = pct_above_ema < 0   # Price below 1d EMA50
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI EXTREME CONDITIONS ===
        rsi_oversold = rsi_7[i] < 25
        rsi_overbought = rsi_7[i] > 75
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: RSI oversold + bull trend + volume spike
            if rsi_oversold and bull_trend and vol_spike:
                desired_signal = SIZE
            
            # SHORT: RSI overbought + bear trend + volume spike
            elif rsi_overbought and bear_trend and vol_spike:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 3 ATR from highest (wider for 12h)
                stop_price = trailing_high - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Take profit: RSI normalizes > 50
                elif rsi_7[i] > 55:
                    desired_signal = SIZE * 0.5  # Half position
                    if rsi_7[i] > 65:
                        desired_signal = 0.0  # Full exit
                        in_position = False
                        position_side = 0
                        
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 3 ATR from lowest
                stop_price = trailing_low + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Take profit: RSI normalizes < 50
                elif rsi_7[i] < 45:
                    desired_signal = -SIZE * 0.5  # Half position
                    if rsi_7[i] < 35:
                        desired_signal = 0.0  # Full exit
                        in_position = False
                        position_side = 0
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals