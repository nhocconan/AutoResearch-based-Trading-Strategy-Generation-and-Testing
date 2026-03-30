#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian(15) + RSI + 1w Trend + Volume (4h)

HYPOTHESIS: Lower timeframe (4h) + looser entry = sufficient trades.
Donchian(15) instead of (20) → more frequent breakouts
RSI(14) > 55 for longs, < 45 for shorts = momentum confirmation
1w close above/below SMA(8) = major trend filter
Volume spike 1.3x = confirmation without being too restrictive

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout above 15-bar high + RSI>55 + 1w>MA8 + vol spike = momentum trade
- Bear: Breakdown below 15-bar low + RSI<45 + 1w<MA8 + vol spike = short momentum
- 4h timeframe = proven to work in DB (358 trades, Sharpe=0.356)

EXPECTED TRADES: 150-300 total over 4 years (37-75/year per symbol)
- Donchian(15) on 4h = break every ~15-30 bars = 328-657 potential/year
- RSI filter (~30% reduction)
- 1w trend filter (~30% reduction)
- Volume spike 1.3x (~40% reduction)
- Final: ~150-300 trades = statistical validity + manageable fees
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian15_rsi_1w_trend_v1"
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
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w close for trend SMA(8)
    weekly_close = df_1w['close'].values
    weekly_sma8 = pd.Series(weekly_close).rolling(window=8, min_periods=8).mean().values
    weekly_sma8_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma8)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian Channel(15) - looser than usual 20
    donchian_upper = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donchian_lower = pd.Series(low).rolling(window=15, min_periods=15).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average (15 bars to match Donchian)
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
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
    
    warmup = 50  # Enough for Donchian15, ATR14, RSI14
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_sma8_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: 1w close vs SMA(8) ===
        bull_trend = close[i] > weekly_sma8_aligned[i]
        bear_trend = close[i] < weekly_sma8_aligned[i]
        
        # === VOLUME CONFIRMATION (1.3x - looser) ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === RSI MOMENTUM FILTER ===
        rsi_val = rsi_14[i] if not np.isnan(rsi_14[i]) else 50.0
        bull_momentum = rsi_val > 55  # Loose: 55 instead of typical 60
        bear_momentum = rsi_val < 45  # Loose: 45 instead of typical 40
        
        # === DONCHIAN BREAKOUT (15 bars) ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + RSI>55 + bull trend + vol spike
            if bullish_breakout and bull_momentum and bull_trend and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + RSI<45 + bear trend + vol spike
            elif bearish_breakout and bear_momentum and bear_trend and vol_spike:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.0 ATR from highest (tightened)
                stop_price = trailing_high - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if RSI loses momentum or trend flips
                elif rsi_val < 45 or close[i] < weekly_sma8_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.0 ATR from lowest (tightened)
                stop_price = trailing_low + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if RSI loses momentum or trend flips
                elif rsi_val > 55 or close[i] > weekly_sma8_aligned[i]:
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