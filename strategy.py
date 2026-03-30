#!/usr/bin/env python3
"""
Experiment #022: Donchian Breakout + RSI Pullback + Volume Spike (4h)

HYPOTHESIS: Simplify to the proven winning formula from DB:
1. Donchian(20) breakout for structure (used in ALL top performers)
2. RSI pullback (<35 long, >65 short) for entry timing
3. Volume spike confirmation for institutional validity
4. 1d SMA200 for trend filter

WHY IT WORKS IN BULL AND BEAR:
- Bull market: Breakout + pullback = classic continuation setup
- Bear market: Breakdown + bounce to RSI 65-70 = short setup
- Range: RSI between 35-65 = no trades, avoids whipsaws

KEY INSIGHT: Top DB performers use ONE signal type + volume + regime.
Complex stacking (Ichimoku + Alligator) = too many filters = 0 trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_rsi_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(data, period=14):
    """RSI with proper min_periods"""
    delta = pd.Series(data).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d SMA200 for trend filter
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=150).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # 1d RSI for momentum filter
    rsi_1d = calculate_rsi(close_1d, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals array
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
    
    warmup = 250  # 20 (donchian) + 14 (RSI) + 200 (HTF SMA200) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND FILTER ===
        htf_uptrend = close[i] > sma200_1d_aligned[i]
        htf_downtrend = close[i] < sma200_1d_aligned[i]
        htf_momentum_bull = rsi_1d_aligned[i] > 50 if not np.isnan(rsi_1d_aligned[i]) else False
        htf_momentum_bear = rsi_1d_aligned[i] < 50 if not np.isnan(rsi_1d_aligned[i]) else False
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Price breaks 20-bar high with volume
        price_broke_high = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
        price_broke_low = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
        
        # RSI pullback confirmation
        rsi_pulled_back = rsi_14[i] < 38  # RSI pulled back = good entry
        rsi_rallied = rsi_14[i] > 62       # RSI rallied = good short entry
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout + RSI pullback + volume + HTF bull
            if price_broke_high and rsi_pulled_back and vol_spike:
                if htf_uptrend and htf_momentum_bull:
                    desired_signal = SIZE
                elif htf_uptrend:  # Allow if 1d trend up even if RSI 1d < 50
                    desired_signal = SIZE * 0.8  # Reduce size for weaker HTF momentum
            
            # SHORT: Breakdown + RSI rallied + volume + HTF bear
            if price_broke_low and rsi_rallied and vol_spike:
                if htf_downtrend and htf_momentum_bear:
                    desired_signal = -SIZE
                elif htf_downtrend:
                    desired_signal = -SIZE * 0.8
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: price must stay above high - 2.5*ATR
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if RSI becomes overbought AND price below SMA200
                if rsi_14[i] > 75 and close[i] < sma200_1d_aligned[i]:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: price must stay below low + 2.5*ATR
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if RSI becomes oversold AND price above SMA200
                if rsi_14[i] < 25 and close[i] > sma200_1d_aligned[i]:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 6 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
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