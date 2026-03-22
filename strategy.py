#!/usr/bin/env python3
"""
Experiment #003: 1h Vol Spike Reversion + 4h HMA Trend + ADX Regime
Hypothesis: Volatility spike reversion (ATR(7)/ATR(30) > 1.5) combined with 
Bollinger Band extremes captures panic selling/buying exhaustion. 
4h HMA provides trend bias, ADX filters choppy vs trending regimes.
Looser entry conditions than CRSI to ensure >=10 trades per symbol on train.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fees.
Stoploss: 2.5*ATR trailing stop to limit drawdown.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Key fix from #001: Much looser entry thresholds to guarantee trade generation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_4h_hma_adx_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx)] = 0
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50.0
    return rsi.values

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
    
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Volatility ratio for spike detection (looser threshold for more trades)
    vol_ratio = atr_7 / atr_30
    vol_ratio[np.isnan(vol_ratio)] = 1.0
    vol_ratio = np.clip(vol_ratio, 0, 10)
    
    # RSI extremes for mean reversion
    rsi_oversold = rsi < 35
    rsi_overbought = rsi > 65
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Volatility spike detection (looser: 1.5x instead of 2.0x)
        vol_spike = vol_ratio[i] > 1.5
        vol_normal = vol_ratio[i] < 1.3
        
        # ADX regime
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # Price position vs Bollinger Bands (looser thresholds for more trades)
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        price_near_lower = close[i] < bb_lower[i] * 1.02
        price_near_upper = close[i] > bb_upper[i] * 0.98
        price_below_sma = close[i] < bb_sma[i]
        price_above_sma = close[i] > bb_sma[i]
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        # === STOPLOSS CHECK FIRST (before new signal) ===
        stoploss_triggered = False
        
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                stoploss_triggered = True
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            continue
        
        # === NEW SIGNAL CALCULATION ===
        new_signal = 0.0
        
        # === LONG ENTRY (multiple paths for more trades) ===
        # Path 1: Vol spike + BB lower + 4h bull (strongest)
        if vol_spike and price_below_lower and bull_trend:
            new_signal = SIZE_MAX
        # Path 2: RSI oversold + 4h bull + weak trend (mean reversion)
        elif rsi_oversold[i] and bull_trend and weak_trend:
            new_signal = SIZE_BASE
        # Path 3: Price below BB lower + EMA bullish
        elif price_below_lower and ema_bullish:
            new_signal = SIZE_BASE
        # Path 4: Vol spike + 4h bull + RSI oversold (simpler)
        elif vol_spike and bull_trend and rsi_oversold[i]:
            new_signal = SIZE_BASE
        # Path 5: Price near BB lower + 4h bull (most frequent)
        elif price_near_lower and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY (multiple paths for more trades) ===
        # Path 1: Vol spike + BB upper + 4h bear (strongest)
        if vol_spike and price_above_upper and bear_trend:
            new_signal = -SIZE_MAX
        # Path 2: RSI overbought + 4h bear + weak trend (mean reversion)
        elif rsi_overbought[i] and bear_trend and weak_trend:
            new_signal = -SIZE_BASE
        # Path 3: Price above BB upper + EMA bearish
        elif price_above_upper and ema_bearish:
            new_signal = -SIZE_BASE
        # Path 4: Vol spike + 4h bear + RSI overbought (simpler)
        elif vol_spike and bear_trend and rsi_overbought[i]:
            new_signal = -SIZE_BASE
        # Path 5: Price near BB upper + 4h bear (most frequent)
        elif price_near_upper and bear_trend:
            new_signal = -SIZE_BASE
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals