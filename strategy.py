#!/usr/bin/env python3
"""
Experiment #259: 15m Supertrend + 4h HMA Trend + Volume Momentum
Hypothesis: 15m Supertrend provides clean entry signals while 4h HMA filters 
the macro trend direction. Volume momentum (taker buy ratio) confirms conviction.
RSI is used as soft confirmation only (not hard filter) to ensure sufficient trades.
This differs from failed RSI pullback strategies by making Supertrend the primary 
trigger, not RSI. Position sizing: 0.25 entry, stoploss at 2.5*ATR trailing.
Target: Beat Sharpe=0.499 with more consistent 15m momentum captures.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_volume_momentum_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend indicator (trend following)."""
    n = len(close)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(n):
        if i == 0:
            upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
            lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
            supertrend[i] = upper_band[i]
            continue
        
        # Calculate basic bands
        prev_upper = upper_band[i-1]
        prev_lower = lower_band[i-1]
        hl_mid = (high[i] + low[i]) / 2
        
        # Upper band: only moves up
        upper_band[i] = max(hl_mid + multiplier * atr[i], prev_upper) if close[i-1] > prev_upper else hl_mid + multiplier * atr[i]
        
        # Lower band: only moves down
        lower_band[i] = min(hl_mid - multiplier * atr[i], prev_lower) if close[i-1] < prev_lower else hl_mid - multiplier * atr[i]
        
        # Determine trend and supertrend value
        if close[i] > prev_upper:
            trend[i] = 1
            supertrend[i] = lower_band[i]
        elif close[i] < prev_lower:
            trend[i] = -1
            supertrend[i] = upper_band[i]
        else:
            trend[i] = trend[i-1]
            supertrend[i] = lower_band[i] if trend[i] == 1 else upper_band[i]
    
    return supertrend, trend

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def calculate_momentum(close, period=10):
    """Calculate price momentum (ROC-like)."""
    mom = np.zeros(len(close))
    for i in range(period, len(close)):
        if close[i-period] > 0:
            mom[i] = (close[i] - close[i-period]) / close[i-period] * 100
        else:
            mom[i] = 0.0
    return mom

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    momentum = calculate_momentum(close, 10)
    
    # Calculate Supertrend
    supertrend, st_trend = calculate_supertrend(high, low, close, atr, 3.0)
    
    # Track previous values for signal changes
    prev_st_trend = np.roll(st_trend, 1)
    prev_st_trend[0] = st_trend[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (4h HMA)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Supertrend signals (primary trigger)
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        st_cross_up = prev_st_trend[i] == -1 and st_trend[i] == 1
        st_cross_down = prev_st_trend[i] == 1 and st_trend[i] == -1
        
        # RSI soft confirmation (not hard filter)
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        rsi_not_extreme = 25 < rsi[i] < 75
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        vol_strong = vol_ratio[i] > 0.55 or vol_ratio[i] < 0.45
        
        # Momentum confirmation
        mom_positive = momentum[i] > 0.5
        mom_negative = momentum[i] < -0.5
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Supertrend cross up with HTF trend (primary signal)
        if st_cross_up:
            if hma_4h_bullish:
                new_signal = SIZE_ENTRY
            elif rsi_bullish and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # Supertrend already bullish + pullback entry
        elif st_bullish and hma_4h_bullish:
            # Price near supertrend support
            if close[i] < supertrend[i] * 1.02 and close[i] > supertrend[i]:
                if rsi_not_extreme or vol_bullish:
                    new_signal = SIZE_ENTRY
        
        # Momentum breakout with trend
        elif mom_positive and hma_4h_bullish:
            if st_bullish and vol_strong:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Supertrend cross down with HTF trend (primary signal)
        if st_cross_down:
            if hma_4h_bearish:
                new_signal = -SIZE_ENTRY
            elif rsi_bearish and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # Supertrend already bearish + pullback entry
        elif st_bearish and hma_4h_bearish:
            # Price near supertrend resistance
            if close[i] > supertrend[i] * 0.98 and close[i] < supertrend[i]:
                if rsi_not_extreme or vol_bearish:
                    new_signal = -SIZE_ENTRY
        
        # Momentum breakdown with trend
        elif mom_negative and hma_4h_bearish:
            if st_bearish and vol_strong:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals