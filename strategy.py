#!/usr/bin/env python3
"""
Experiment #005: 1d Donchian Breakout + 1w EMA21 + Volume Spike

HYPOTHESIS: 1d Donchian(20) captures ~monthly volatility swings.
By requiring BOTH a 1w EMA21 trend alignment AND volume confirmation on the break,
we filter out false breakouts while maintaining enough signals.

WHY 1d + 1w: 1w EMA21 = ~5 months of weekly data = major trend direction.
Donchian(20) on 1d = 20 trading days = ~monthly channel.
Entry only when price breaks channel WITH trend confirmation + volume spike.

KEY FIX FROM PREVIOUS FAILURES: Previous 1d Donchian had only 22 trades (too few).
Using 1w EMA instead of 1d SMA gives stronger signal. Donchian break WITH both
trend and volume = tighter entry = better Sharpe despite fewer total trades.

TARGET: 75-150 total over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_ema_v1"
timeframe = "1d"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA21 for major trend direction (~5 months)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20-period = ~monthly)
    upper_donchian = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_donchian = (upper_donchian + lower_donchian) / 2.0
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Need enough for Donchian(20) + ATR(14)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(upper_donchian[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA21) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        upper_break = high[i] >= upper_donchian[i]  # Break above 20d high
        lower_break = low[i] <= lower_donchian[i]   # Break below 20d low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper Donchian break + uptrend + volume ===
            if upper_break and price_above_1w_ema and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Lower Donchian break + downtrend + volume ===
            if lower_break and not price_above_1w_ema and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                stop_loss = entry_price - 2.5 * entry_atr
                if low[i] < stop_loss:
                    desired_signal = 0.0
                # Take profit at mid-Donchian if reached
                elif close[i] >= mid_donchian[i] and bars_held >= 3:
                    desired_signal = 0.0
                    
            if position_side < 0:
                stop_loss = entry_price + 2.5 * entry_atr
                if high[i] > stop_loss:
                    desired_signal = 0.0
                # Take profit at mid-Donchian if reached
                elif close[i] <= mid_donchian[i] and bars_held >= 3:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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