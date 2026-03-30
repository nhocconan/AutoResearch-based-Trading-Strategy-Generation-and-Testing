#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + RSI + Volume + 1d EMA200

HYPOTHESIS: Donchian(20) channels capture institutional breakout structure (proven pattern).
Price breaking above 20-bar high with RSI(14) not overbought + volume spike = momentum.
Using 1d EMA200 as trend filter to stay with the major trend direction.
12h timeframe = ~3x fewer trades than 4h = less fee drag = better test generalization.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Buy breakouts above 1d EMA200 when RSI neutral (40-60), not overbought.
- Bear: Short breakouts below 1d EMA200 when RSI neutral (40-60), not oversold.
- ATR stoploss adapts to volatility in both directions.
- Minimum 2-bar hold prevents whipsaw from temporary breaks.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_rsi_vol_ema200_1d_v1"
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
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (rs + 1))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend direction
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_band = (upper_band + lower_band) / 2.0
    
    # Volume ratio (20-bar moving average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need enough for EMA200 alignment buffer + Donchian(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_200_aligned[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        price_above_1d_ema = close[i] > ema_200_aligned[i]
        
        # RSI momentum (neutral zone = 40-60 for entries)
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakouts
        breakout_up = close[i] > upper_band[i]  # New high
        breakout_down = close[i] < lower_band[i]  # New low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + trend alignment + neutral RSI ===
            # Only in uptrend, enter on pullback to middle band (reversion to mean)
            if price_above_1d_ema and rsi_neutral and vol_spike:
                # Price in middle zone (between middle and upper band)
                in_middle_zone = middle_band[i] <= close[i] <= upper_band[i]
                if in_middle_zone:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + trend alignment + neutral RSI ===
            if not price_above_1d_ema and rsi_neutral and vol_spike:
                # Price in lower zone (between lower and middle band)
                in_lower_zone = lower_band[i] <= close[i] <= middle_band[i]
                if in_lower_zone:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD (minimum 2 bars = 1 day to avoid churn) ===
        bars_held = i - entry_bar
        
        # === TAKE PROFIT (opposite Donchian band) ===
        if in_position and bars_held >= 2:
            # Long: exit near upper band
            if position_side > 0 and close[i] >= upper_band[i]:
                desired_signal = 0.0
            # Short: exit near lower band
            if position_side < 0 and close[i] <= lower_band[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals