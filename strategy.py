#!/usr/bin/env python3
"""
Experiment #024: 4h ATR Channel Breakout + 1d SMA200 + Volume Spike

HYPOTHESIS: ATR-channel breakout adapts to volatility better than fixed-% Donchian.
In high-vol environments (2022 crash), the wider channel prevents false breakouts.
Combined with 1d SMA200 trend filter for direction and tight ATR-based stops.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Breakout above ATR channel = momentum continuation, ride the rally
- Bear: Breakout below ATR channel = short rallies, avoid long false breakouts
- ATR stop adapts to vol: 2*ATR in crash = wider stop = fewer stop-outs on wicks
- 1d SMA200 catches major trend changes (2022 crash = price crosses below = shorts)

Previous experiments to avoid:
- Elder Ray (failed 012, 020)
- Fixed % Donchian (too few trades)
- KAMA-only trend (005 = 0 trades)
- 1d timeframe (crashes too slow)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_atr_channel_1d_sma200_vol_v1"
timeframe = "4h"
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

def calculate_donchian(high, low, period):
    """Donchian Channel: returns (upper, lower, middle)"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_bull_bear_power(high, low, close, ema_period=13):
    """Elder Ray Bull/Bear Power"""
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_aligned if (sma_200_aligned := sma_200_1d) else sma_200_1d)
    
    # === Pre-compute all indicators BEFORE loop ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR-based channel (adapts to volatility)
    # Channel width = ATR(14) * multiplier
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    channel_mult_upper = 0.5  # Upper channel at close + 0.5 * ATR_ma
    channel_mult_lower = 0.5  # Lower channel at close - 0.5 * ATR_ma
    
    upper_channel = close + channel_mult_upper * atr_ma
    lower_channel = close - channel_mult_lower * atr_ma
    
    # Donchian(20) for structure
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Bull/Bear power for exit
    bull_power, bear_power = calculate_bull_bear_power(high, low, close, ema_period=13)
    bull_smooth = pd.Series(bull_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    bear_smooth = pd.Series(bear_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # RSI for exit
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% of capital
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 220  # 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0 or np.isnan(atr_ma[i]) or atr_ma[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        
        # === ATR-normalized Elder Ray for confirmation ===
        atr_local = atr_14[i]
        bull_norm = bull_smooth[i] / atr_local
        bear_norm = bear_smooth[i] / atr_local
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Long: Price breaks above ATR channel AND above 1d SMA AND bull power positive
            # ATR channel breakout: close > upper_channel (ATR-adaptive)
            long_breakout = close[i] > upper_channel[i]
            if long_breakout and price_above_1d_sma and bull_norm > 0 and vol_spike:
                desired_signal = SIZE
            
            # Short: Price breaks below ATR channel AND below 1d SMA AND bear power negative
            short_breakout = close[i] < lower_channel[i]
            if short_breakout and not price_above_1d_sma and bear_norm < 0 and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD TIME (4 bars = 16 hours) ===
        bars_held = i - entry_bar if in_position else 0
        min_hold_bars = 4
        
        if in_position and bars_held >= min_hold_bars:
            # Exit if Elder Ray reverses (momentum shift)
            if position_side > 0 and bear_norm < -0.3:
                desired_signal = 0.0
            if position_side < 0 and bull_norm > 0.3:
                desired_signal = 0.0
        
        # === RSI EXIT FILTER ===
        if in_position:
            if position_side > 0 and rsi[i] > 75:
                desired_signal = 0.0
            if position_side < 0 and rsi[i] < 25:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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
        
        signals[i] = desired_signal
    
    return signals