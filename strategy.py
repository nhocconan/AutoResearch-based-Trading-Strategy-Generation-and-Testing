#!/usr/bin/env python3
"""
Experiment #016: 4h Volatility Mean Reversion + 1d Trend Filter
Hypothesis: Volatility spikes (ATR ratio > 1.8) + price at Bollinger extremes create 
high-probability mean reversion setups. 1d HMA provides trend bias to avoid 
counter-trend trades in strong trends. This targets the "vol crush" pattern after 
panic selling, which research shows works well in bear markets (2022 crash, 2025 test).

Key innovations:
1. ATR(7)/ATR(30) ratio detects volatility expansion (panic spikes)
2. Bollinger %B confirms price at extremes (<0.05 or >0.95)
3. 1d HMA trend filter avoids counter-trend in strong moves
4. Conservative position sizing (0.25-0.30) with 2*ATR trailing stop
5. Relaxed entry thresholds to ensure ≥10 trades/symbol on train

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (get_htf_data called ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_meanrev_1d_hma_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi[np.isnan(rsi)] = 50.0
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and %B indicator."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # %B: where price is within bands (0=lower, 1=upper, >1=above, <0=below)
    band_range = upper - lower
    band_range[band_range == 0] = 1e-10
    percent_b = (close - lower) / band_range
    percent_b[np.isnan(percent_b)] = 0.5
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, percent_b, bandwidth, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std.replace(0, np.nan)
    zscore = zscore.values
    zscore[np.isnan(zscore)] = 0.0
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # ATR ratio for volatility expansion detection
    atr_ratio = atr_7 / atr_30
    atr_ratio[np.isnan(atr_ratio) | (atr_30 == 0)] = 1.0
    atr_ratio = np.clip(atr_ratio, 0.5, 5.0)
    
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    
    bb_upper, bb_lower, percent_b, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    zscore_20 = calculate_zscore(close, 20)
    
    # EMA for additional trend confirmation
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(percent_b[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF)
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # Volatility expansion detection (relaxed threshold for more trades)
        vol_spike = atr_ratio[i] > 1.6  # Lowered from 2.0 for more signals
        vol_extreme = atr_ratio[i] > 2.0
        
        # Bollinger %B extremes (price at band edges)
        price_at_lower = percent_b[i] < 0.10  # Near or below lower band
        price_at_upper = percent_b[i] > 0.90  # Near or above upper band
        price_extreme_lower = percent_b[i] < 0.05
        price_extreme_upper = percent_b[i] > 0.95
        
        # RSI extremes (relaxed for more trades)
        rsi_oversold = rsi_14[i] < 35  # Lowered from 30
        rsi_overbought = rsi_14[i] > 65  # Lowered from 70
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_extreme_overbought = rsi_14[i] > 75
        
        # Z-score extremes
        zscore_extreme_low = zscore_20[i] < -1.8
        zscore_extreme_high = zscore_20[i] > 1.8
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: RSI oversold + price at lower BB + vol spike + 1d bull trend
        if rsi_oversold and price_at_lower and vol_spike and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: RSI extreme oversold + price extreme lower + 1d bull trend
        elif rsi_extreme_oversold and price_extreme_lower and bull_trend:
            new_signal = SIZE_MAX
        # Tertiary: Z-score extreme low + price at lower BB + vol spike
        elif zscore_extreme_low and price_at_lower and vol_spike:
            new_signal = SIZE_BASE
        # Quaternary: RSI oversold + price at lower BB (no trend filter for range)
        elif rsi_oversold and price_extreme_lower and not bear_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: RSI overbought + price at upper BB + vol spike + 1d bear trend
        if rsi_overbought and price_at_upper and vol_spike and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: RSI extreme overbought + price extreme upper + 1d bear trend
        elif rsi_extreme_overbought and price_extreme_upper and bear_trend:
            new_signal = -SIZE_MAX
        # Tertiary: Z-score extreme high + price at upper BB + vol spike
        elif zscore_extreme_high and price_at_upper and vol_spike:
            new_signal = -SIZE_BASE
        # Quaternary: RSI overbought + price at upper BB (no trend filter for range)
        elif rsi_overbought and price_extreme_upper and not bull_trend:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = highest_close - 2.0 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = lowest_close + 2.0 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr_14[i] if position_side > 0 else close[i] + 2.0 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr_14[i] if position_side > 0 else close[i] + 2.0 * atr_14[i]
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