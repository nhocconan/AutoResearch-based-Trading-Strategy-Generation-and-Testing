#!/usr/bin/env python3
"""
Experiment #015: 1h Vol Spike Mean Reversion with 4h HMA Trend Filter
Hypothesis: Volatility spikes (ATR ratio > 2.5) + Bollinger Band extremes capture
panic bottoms and FOMO tops. 4h HMA provides trend bias to avoid catching falling knives.
This works in 2022 crash (vol spikes at bottoms) and 2025 bear market (range mean reversion).
Key insight from failures: Simple trend following gets whipsawed, pure mean reversion gets run over.
Vol spike filter ensures we only trade extreme conditions (20-50 trades/year target).
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.30 discrete, stoploss 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_bb_reversion_4h_hma_v1"
timeframe = "1h"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    middle = sma
    return upper, middle, lower

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.5)
    
    rsi_14 = calculate_rsi(close, 14)
    zscore_20 = calculate_zscore(close, 20)
    
    # EMA for additional trend filter
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss and take profit
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    take_profit_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(zscore_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility spike detection (ATR ratio)
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 2.0
        vol_normal = atr_ratio < 1.3
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Bollinger Band position
        price_at_lower = close[i] <= bb_lower[i]
        price_at_upper = close[i] >= bb_upper[i]
        price_at_middle = close[i] > bb_middle[i]
        
        # RSI extremes
        rsi_oversold = rsi_14[i] < 25
        rsi_overbought = rsi_14[i] > 75
        
        # Z-score extremes
        zscore_extreme_low = zscore_20[i] < -2.0
        zscore_extreme_high = zscore_20[i] > 2.0
        
        # EMA trend
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Vol spike + price at BB lower + RSI oversold + 4h HMA not strongly bearish
        if vol_spike and price_at_lower and rsi_oversold:
            # Only long if 4h trend is bullish OR we're in extreme oversold (counter-trend)
            if hma_4h_bullish or (zscore_extreme_low and rsi_14[i] < 20):
                new_signal = SIZE_ENTRY
        
        # Vol spike + Z-score extreme low (panic bottom)
        elif vol_spike and zscore_extreme_low and rsi_14[i] < 30:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Vol spike + price at BB upper + RSI overbought + 4h HMA not strongly bullish
        if vol_spike and price_at_upper and rsi_overbought:
            # Only short if 4h trend is bearish OR we're in extreme overbought (counter-trend)
            if hma_4h_bearish or (zscore_extreme_high and rsi_14[i] > 80):
                new_signal = -SIZE_ENTRY
        
        # Vol spike + Z-score extreme high (FOMO top)
        elif vol_spike and zscore_extreme_high and rsi_14[i] > 70:
            new_signal = -SIZE_ENTRY
        
        # === TAKE PROFIT ===
        # Reduce position by half when price returns to BB middle (1R profit)
        if position_side > 0 and not take_profit_hit and price_at_middle:
            new_signal = SIZE_HALF if new_signal == 0.0 else new_signal
            take_profit_hit = True
        
        if position_side < 0 and not take_profit_hit and price_at_middle:
            new_signal = -SIZE_HALF if new_signal == 0.0 else new_signal
            take_profit_hit = True
        
        # === STOPLOSS LOGIC ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Exit when vol normalizes (mean reversion complete)
        if vol_normal and position_side != 0:
            new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            take_profit_hit = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            take_profit_hit = False
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            take_profit_hit = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            take_profit_hit = True
        
        signals[i] = new_signal
    
    return signals