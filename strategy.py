#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian Breakout + RSI Regime + Daily SMA Trend (4h)

HYPOTHESIS: Combine proven 4h Donchian breakout with RSI as regime filter
instead of complex volume/weekly VWAP logic that failed in #024.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout above Donchian(20) high + price>SMA200(1d) + RSI>50 = momentum continuation
- Bear: Breakdown below Donchian(20) low + price<SMA200(1d) + RSI<50 = momentum continuation
- RSI>50 acts as regime filter: only trend-follow when momentum agrees
- Donchian(20) on 4h = ~80 potential breakouts/year per symbol

LEARNED FROM FAILURES:
- #024 weekly VWAP filter too restrictive → negative Sharpe
- Volume spike requirement too strict → may reduce valid signals
- 6h timeframe showed negative Sharpe (-0.348)
- DB winners use simple 2-3 conditions, not 4+

EXPECTED TRADES: 100-200 total over 4 years (25-50/year per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_rsi_regime_sma200_v1"
timeframe = "4h"
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
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily SMA(200) for trend - aligned to 4h
    daily_close = df_1d['close'].values
    daily_sma200 = pd.Series(daily_close).rolling(window=200, min_periods=200).mean().values
    sma200_aligned = align_htf_to_ltf(prices, df_1d, daily_sma200)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian Channel(20) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 100  # Enough for Donchian20, ATR14, RSI14, SMA200(1d) alignment
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: Daily SMA200 ===
        bull_trend = close[i] > sma200_aligned[i]
        bear_trend = close[i] < sma200_aligned[i]
        
        # === RSI REGIME ===
        bull_momentum = rsi_14[i] > 50
        bear_momentum = rsi_14[i] < 50
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + bull trend + bull momentum
            if bullish_breakout and bull_trend and bull_momentum:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + bear trend + bear momentum
            elif bearish_breakout and bear_trend and bear_momentum:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 3 ATR from highest
                stop_price = trailing_high - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend/momentum flips
                elif close[i] < sma200_aligned[i] * 0.995 or rsi_14[i] < 40:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 3 ATR from lowest
                stop_price = trailing_low + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend/momentum flips
                elif close[i] > sma200_aligned[i] * 1.005 or rsi_14[i] > 60:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
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