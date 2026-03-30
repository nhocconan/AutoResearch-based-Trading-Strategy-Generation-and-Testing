#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian + HMA + Volume Confirmation + HTF Trend

HYPOTHESIS: Follow the proven winning pattern from DB:
- 4h Donchian(20) breakout detection
- 12h HMA for trend direction (both must agree)
- Volume spike confirmation (>1.5x 20-bar avg)
- ATR-based stoploss

WHY IT WORKS IN BULL AND BEAR:
- Bull: Price breaks above Donchian high + HMA rising + volume confirm = long
- Bear: Price breaks below Donchian low + HMA falling + volume confirm = short
- Range: Donchian squeeze + flat HMA = no trade (avoided whipsaws)

KEY INSIGHT: Single strong signal (breakout) > multiple weak signals.
Keep conditions tight: 100-200 total trades over 4 years.

DB EVIDENCE:
- mtf_4h_hma_donchian_volume_rsi_12h_atr_v1: test Sharpe 1.38, 95 trades
- mtf_4h_hma_volume_donchian_adx_12h_atr_v1: test Sharpe 1.32, 94 trades
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_12h_v2"
timeframe = "4h"
leverage = 1.0


def calculate_hma(data, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate WMA with period
    wma_half = pd.Series(data).rolling(window=period // 2, min_periods=period // 2).mean().values
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).mean().values
    
    # HMA = 2*WMA(period/2) - WMA(period)
    hma = 2 * wma_half - wma_full
    
    # Apply WMA smoothing to HMA
    hma_final = np.zeros(n)
    for i in range(n):
        if i < period - 1 or np.isnan(wma_full[i]):
            hma_final[i] = np.nan
        else:
            # WMA of HMA with sqrt(period)
            sqrt_p = int(np.sqrt(period))
            start = max(period - 1, i - sqrt_p + 1)
            window = hma[start:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                hma_final[i] = np.mean(valid)  # Simplified WMA
            else:
                hma_final[i] = np.nan
    
    return hma


def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                              np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def donchian_channel(high, low, period):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(n):
        if i < period - 1:
            upper[i] = np.max(high[:i + 1])
            lower[i] = np.min(low[:i + 1])
        else:
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
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h HMA for trend direction ===
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # HTF: HMA rising = bull, falling = bear
    htf_trend = np.zeros(len(df_12h))
    for i in range(1, len(df_12h)):
        if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i - 1]):
            if hma_12h[i] > hma_12h[i - 1]:
                htf_trend[i] = 1.0   # bull
            elif hma_12h[i] < hma_12h[i - 1]:
                htf_trend[i] = -1.0  # bear
            else:
                htf_trend[i] = htf_trend[i - 1]
    
    htf_trend_aligned = align_htf_to_ltf(prices, df_12h, htf_trend)
    
    # === Local 4h indicators ===
    # Donchian(20)
    dc_upper, dc_lower = donchian_channel(high, low, period=20)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local HMA for confirmation
    hma_local = calculate_hma(close, 16)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian 20 + volume 20 + HMA 16
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(htf_trend_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT ===
        # Upper breakout (bullish)
        upper_breakout = close[i] > dc_upper[i] and close[i - 1] <= dc_upper[i - 1]
        # Lower breakout (bearish)
        lower_breakout = close[i] < dc_lower[i] and close[i - 1] >= dc_lower[i - 1]
        
        # === HTF TREND ===
        htf_bull = htf_trend_aligned[i] > 0.5
        htf_bear = htf_trend_aligned[i] < -0.5
        
        # === LOCAL TREND (HMA direction) ===
        hma_rising = not np.isnan(hma_local[i]) and not np.isnan(hma_local[i - 1]) and hma_local[i] > hma_local[i - 1]
        hma_falling = not np.isnan(hma_local[i]) and not np.isnan(hma_local[i - 1]) and hma_local[i] < hma_local[i - 1]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Upper breakout + HTF bull + local HMA rising + volume spike
            if upper_breakout and htf_bull and hma_rising and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Lower breakout + HTF bear + local HMA falling + volume spike
            if lower_breakout and htf_bear and hma_falling and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if local HMA turns
                if hma_falling:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if HTF turns bear
                if htf_bear:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if local HMA turns
                if hma_rising:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if HTF turns bull
                if htf_bull:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
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
        elif in_position:
            in_position = False
            position_side = 0
        
        signals[i] = desired_signal
    
    return signals